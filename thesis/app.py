import streamlit as st
import requests
from urllib.parse import urlencode
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize session state
if 'eventbrite_token' not in st.session_state:
    st.session_state.eventbrite_token = None

AUTH_URL = "https://www.eventbrite.com/oauth/authorize"
TOKEN_URL = "https://www.eventbrite.com/oauth/token"

def handle_oauth():
    #query_params = st.experimental_get_query_params()
    query_params = st.query_params
    # Step 1: Redirect to Eventbrite if no code/token
    if not st.session_state.eventbrite_token and 'code' not in query_params:
        auth_params = {
            'response_type': 'code',
            'client_id': os.getenv('EVENTBRITE_CLIENT_ID'),
            'redirect_uri': os.getenv('EVENTBRITE_REDIRECT_URI')
        }
        auth_request = requests.Request('GET', AUTH_URL, params=auth_params).prepare()
        st.markdown(f"[Authorize with Eventbrite]({auth_request.url})")
        st.stop()
    
    # Step 2: Exchange code for token
    if 'code' in query_params and not st.session_state.eventbrite_token:
        try:
            code = query_params['code'][0]
            token_data = {
                'grant_type': 'authorization_code',
                'client_id': os.getenv('EVENTBRITE_CLIENT_ID'),
                'client_secret': os.getenv('EVENTBRITE_CLIENT_SECRET'),
                'code': code,
                'redirect_uri': os.getenv('EVENTBRITE_REDIRECT_URI')
            }
            
            response = requests.post(TOKEN_URL, data=token_data)
            response.raise_for_status()
            
            st.session_state.eventbrite_token = response.json()['access_token']
            st.experimental_set_query_params()
            st.rerun()
            
        except Exception as e:
            st.error(f"Authorization failed: {str(e)}")
            st.session_state.eventbrite_token = None

def make_api_request(endpoint):
    if not st.session_state.eventbrite_token:
        st.error("Not authenticated!")
        st.stop()
        
    headers = {"Authorization": f"Bearer {st.session_state.eventbrite_token}"}
    response = requests.get(f"https://www.eventbriteapi.com/v3/{endpoint}", headers=headers)
    return response

# Main app flow
handle_oauth()

if st.session_state.eventbrite_token:
    st.title("Authorized App")
    
    # Example API call
    try:
        response = make_api_request("users/me/")
        if response.status_code == 200:
            user_data = response.json()
            st.write(f"Welcome, {user_data['name']}!")
        else:
            st.error(f"API request failed: {response.text}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
else:
    st.warning("Authentication required")
