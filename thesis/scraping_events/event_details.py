import pandas as pd
import numpy as np
import random
from datetime import datetime
import string

# Function to load your events data
def load_events_data(file_path):
    events_df = pd.read_csv(file_path)
    return events_df

# Define event categories - these will be assigned based on patterns rather than specific words
EVENT_CATEGORIES = [
    'Music & Concerts', 
    'Food & Drink', 
    'Education & Learning', 
    'Sports & Fitness', 
    'Arts & Culture', 
    'Business & Networking', 
    'Technology', 
    'Community & Causes', 
    'Health & Wellness', 
    'Entertainment'
]

# Function to generate placeholder words for event content generation
def generate_category_vocabularies():
    """
    Creates category-specific vocabularies for generating content
    when we don't know the actual word stems
    """
    return {
        'Music & Concerts': {
            'nouns': ['concert', 'festival', 'performance', 'band', 'artist', 'stage', 'tour', 'album', 'show', 'music'],
            'adjectives': ['live', 'acoustic', 'electric', 'annual', 'sold-out', 'underground', 'trending', 'indie'],
            'verbs': ['perform', 'present', 'feature', 'showcase', 'introduce', 'celebrate', 'launch'],
            'genres': ['rock', 'jazz', 'classical', 'hip-hop', 'electronic', 'folk', 'blues', 'country', 'pop', 'r&b']
        },
        'Food & Drink': {
            'nouns': ['festival', 'tasting', 'dinner', 'brunch', 'chef', 'restaurant', 'food', 'cuisine', 'menu', 'flavor'],
            'adjectives': ['gourmet', 'craft', 'artisanal', 'local', 'farm-to-table', 'international', 'authentic', 'organic'],
            'verbs': ['taste', 'sample', 'savor', 'pair', 'enjoy', 'experience', 'discover', 'share'],
            'cuisines': ['Italian', 'Mexican', 'Asian', 'Mediterranean', 'American', 'French', 'Indian', 'Thai', 'Japanese']
        },
        # Other categories would follow the same pattern...
        'Education & Learning': {
            'nouns': ['workshop', 'seminar', 'class', 'course', 'lecture', 'training', 'certification', 'symposium'],
            'adjectives': ['interactive', 'hands-on', 'beginner', 'advanced', 'professional', 'certified', 'accredited'],
            'verbs': ['learn', 'master', 'develop', 'discover', 'explore', 'understand', 'practice', 'train'],
            'topics': ['programming', 'design', 'marketing', 'finance', 'leadership', 'writing', 'photography', 'language']
        },
        'Sports & Fitness': {
            'nouns': ['tournament', 'game', 'match', 'race', 'competition', 'league', 'challenge', 'championship'],
            'adjectives': ['competitive', 'amateur', 'professional', 'local', 'regional', 'annual', 'seasonal', 'intense'],
            'verbs': ['compete', 'play', 'run', 'join', 'participate', 'challenge', 'train', 'race'],
            'sports': ['basketball', 'soccer', 'tennis', 'running', 'cycling', 'yoga', 'crossfit', 'swimming']
        },
        'Arts & Culture': {
            'nouns': ['exhibition', 'gallery', 'showing', 'collection', 'installation', 'display', 'artwork', 'museum'],
            'adjectives': ['contemporary', 'traditional', 'innovative', 'cultural', 'artistic', 'creative', 'visual', 'immersive'],
            'verbs': ['present', 'showcase', 'exhibit', 'feature', 'display', 'celebrate', 'explore', 'represent'],
            'art_forms': ['painting', 'sculpture', 'photography', 'film', 'mixed media', 'digital art', 'pottery', 'textiles']
        },
        'Business & Networking': {
            'nouns': ['conference', 'meetup', 'networking', 'summit', 'forum', 'roundtable', 'workshop', 'session'],
            'adjectives': ['professional', 'executive', 'industry', 'corporate', 'entrepreneurial', 'strategic', 'innovative'],
            'verbs': ['connect', 'network', 'discuss', 'exchange', 'build', 'develop', 'grow', 'learn'],
            'fields': ['marketing', 'technology', 'finance', 'healthcare', 'retail', 'startup', 'consulting', 'leadership']
        },
        'Technology': {
            'nouns': ['hackathon', 'demo', 'conference', 'workshop', 'meetup', 'launch', 'talk', 'session'],
            'adjectives': ['digital', 'innovative', 'cutting-edge', 'tech', 'emerging', 'disruptive', 'smart', 'next-gen'],
            'verbs': ['code', 'develop', 'launch', 'present', 'demonstrate', 'build', 'create', 'innovate'],
            'tech_areas': ['AI', 'blockchain', 'data science', 'web development', 'mobile apps', 'cloud computing', 'cybersecurity']
        },
        'Community & Causes': {
            'nouns': ['fundraiser', 'drive', 'volunteer', 'charity', 'benefit', 'community', 'initiative', 'cause'],
            'adjectives': ['local', 'nonprofit', 'charitable', 'grassroots', 'community-led', 'impactful', 'sustainable'],
            'verbs': ['support', 'help', 'contribute', 'join', 'participate', 'volunteer', 'donate', 'empower'],
            'causes': ['environmental', 'educational', 'healthcare', 'housing', 'hunger', 'equality', 'youth', 'elderly']
        },
        'Health & Wellness': {
            'nouns': ['retreat', 'session', 'class', 'workshop', 'program', 'practice', 'clinic', 'therapy'],
            'adjectives': ['holistic', 'mindful', 'therapeutic', 'wellness', 'rejuvenating', 'balanced', 'natural', 'healing'],
            'verbs': ['practice', 'improve', 'restore', 'heal', 'learn', 'nourish', 'balance', 'strengthen'],
            'practices': ['yoga', 'meditation', 'fitness', 'nutrition', 'mindfulness', 'massage', 'therapy', 'wellness']
        },
        'Entertainment': {
            'nouns': ['show', 'performance', 'screening', 'premiere', 'night', 'tour', 'event', 'party'],
            'adjectives': ['live', 'exclusive', 'special', 'featured', 'entertaining', 'unforgettable', 'premier', 'anticipated'],
            'verbs': ['present', 'showcase', 'feature', 'perform', 'entertain', 'premiere', 'host', 'celebrate'],
            'genres': ['comedy', 'theater', 'film', 'dance', 'variety', 'improv', 'stand-up', 'drama']
        }
    }

# Define title and description templates for each category
def get_templates():
    """
    Get templates for generating titles and descriptions for each category
    """
    title_templates = {
        'Music & Concerts': [
            "{adjective} {noun} in {city}",
            "{genre} {noun}: {season} Series",
            "{city} {genre} {noun}",
            "{adjective} {genre} {noun}"
        ],
        'Food & Drink': [
            "{city} {adjective} {noun}",
            "{cuisine} {noun} {season}",
            "{adjective} {noun} in {city}",
            "{season} {cuisine} {noun}"
        ],
        # Other categories would follow the same pattern
        'Education & Learning': [
            "{topic} {noun}: {adjective} {verb}ing",
            "{adjective} {topic} {noun}",
            "{verb} {topic}: {adjective} {noun}",
            "{city} {topic} {noun}"
        ],
        'Sports & Fitness': [
            "{city} {sport} {noun}",
            "{adjective} {sport} {noun}",
            "{season} {sport} {noun}",
            "{sport} {noun}: {adjective} {verb}"
        ],
        'Arts & Culture': [
            "{adjective} {art_form} {noun}",
            "{art_form} {noun}: {adjective} {verb}",
            "{city} {art_form} {noun}",
            "{adjective} {noun}: {art_form} {verb}ing"
        ],
        'Business & Networking': [
            "{field} {noun}: {adjective} {verb}ing",
            "{city} {field} {noun}",
            "{adjective} {noun} for {field} Professionals",
            "{season} {field} {noun}"
        ],
        'Technology': [
            "{tech_area} {noun}",
            "{adjective} {tech_area} {noun}",
            "{city} {tech_area} {noun}",
            "{verb}ing {tech_area}: {adjective} {noun}"
        ],
        'Community & Causes': [
            "{city} {cause} {noun}",
            "{adjective} {cause} {noun}",
            "{verb} for {cause}: {noun}",
            "{season} {cause} {noun}"
        ],
        'Health & Wellness': [
            "{adjective} {practice} {noun}",
            "{practice} {noun}: {adjective} {verb}ing",
            "{city} {practice} {noun}",
            "{season} {practice} {noun}"
        ],
        'Entertainment': [
            "{adjective} {genre} {noun}",
            "{genre} {noun} in {city}",
            "{season} {genre} {noun}",
            "{city} {genre}: {adjective} {noun}"
        ]
    }
    
    description_templates = {
        'Music & Concerts': [
            "Join us for an {adjective} {genre} {noun} in {city}. This {season} event will feature {adjective} performances that you won't want to miss. {time_info}",
            "Experience {adjective} {genre} music at our {city} venue. This {noun} brings together talented artists for a night of unforgettable entertainment. {time_info}",
            "The {season} {genre} {noun} returns to {city}! {verb} with us as we celebrate music with performances throughout the event. {time_info}"
        ],
        'Food & Drink': [
            "Indulge in {adjective} {cuisine} cuisine at this special {noun} in {city}. Perfect for food enthusiasts looking to {verb} new flavors. {time_info}",
            "Join us for a celebration of {adjective} {cuisine} food and drinks. Located in {city}, this {season} {noun} offers something for everyone. {time_info}",
            "Experience a unique culinary journey with {cuisine} specialties at our {adjective} {noun}. Our {city} venue will host local and visiting chefs. {time_info}"
        ],
        # Other categories would follow the same pattern
        'Education & Learning': [
            "Expand your knowledge on {topic} with our {adjective} {noun}. Suitable for all experience levels looking to {verb} new skills. {time_info}",
            "Learn practical {topic} skills through this {adjective} {noun}. Hosted by experts in the field, you'll {verb} through hands-on activities. {time_info}",
            "Our {adjective} {topic} {noun} offers in-depth training for professionals. Certificates will be provided upon completion. {time_info}"
        ],
        'Sports & Fitness': [
            "Join the {adjective} {sport} community for our {season} {noun} in {city}. Athletes of all levels are welcome to {verb} in this exciting event. {time_info}",
            "Challenge yourself at the {city} {sport} {noun}. This {adjective} event will test your skills and endurance. {time_info}",
            "The {season} {sport} {noun} is back! {verb} with fellow enthusiasts in this {adjective} community event. {time_info}"
        ]
        # Other categories would have similar templates
    }
    
    # Add missing templates for categories not explicitly defined
    for category in EVENT_CATEGORIES:
        if category not in title_templates:
            title_templates[category] = [
                "{adjective} {noun} in {city}",
                "{city} {adjective} {noun}",
                "{season} {noun}: {adjective} Event",
                "{adjective} {noun}: {city} Edition"
            ]
        
        if category not in description_templates:
            description_templates[category] = [
                "Join us for this {adjective} {noun} in {city}. This {season} event promises to be an exciting opportunity to {verb} with others in the community. {time_info}",
                "Don't miss our {adjective} {noun} happening in {city}. This is your chance to {verb} and connect with like-minded individuals. {time_info}",
                "We're excited to bring you this {season} {noun} in {city}. Come and {verb} in this {adjective} community event. {time_info}"
            ]
    
    return title_templates, description_templates

# Function to determine category based on count patterns rather than specific words
def determine_category(event_row):
    """
    Determine event category based on frequency distribution patterns
    rather than actual word meanings
    """
    # Extract count values
    count_columns = [col for col in event_row.index if col.startswith('count_') and col != 'count_other']
    counts = [event_row[col] for col in count_columns]
    
    if not counts:  # If no counts available
        return random.choice(EVENT_CATEGORIES)
    
    # Analyze patterns in the counts
    total_words = sum(counts) + event_row.get('count_other', 0)
    num_unique_words = sum(1 for count in counts if count > 0)
    max_count = max(counts) if counts else 0
    concentration = max_count / total_words if total_words > 0 else 0
    
    # Simple heuristic categorization based on word distribution patterns
    # (Without knowing actual words, this is approximate)
    if num_unique_words > 15 and concentration < 0.15:
        # Diverse vocabulary, lower concentration - likely educational or business
        return random.choice(['Education & Learning', 'Business & Networking', 'Technology'])
    elif concentration > 0.3:
        # High concentration on few words - likely focused events like concerts or sports
        return random.choice(['Music & Concerts', 'Sports & Fitness', 'Entertainment'])
    elif 0.2 <= concentration <= 0.3 and num_unique_words < 10:
        # Medium concentration, fewer unique words - likely food or arts
        return random.choice(['Food & Drink', 'Arts & Culture', 'Health & Wellness'])
    else:
        # Default to random selection with probability weights
        return random.choices(
            EVENT_CATEGORIES,
            weights=[0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.05],
            k=1
        )[0]

# Create a function to format the time
def format_time(time_str):
    """Format ISO time string to readable time"""
    if pd.isna(time_str):
        return "TBD"
    
    try:
        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        return dt.strftime("%A, %B %d at %-I:%M %p")  # e.g., "Monday, January 15 at 7:30 PM"
    except:
        return "TBD"

# Function to generate a title
def generate_title(category, event_row, vocabularies, title_templates):
    """Generate an event title based on category and patterns"""
    # Select a random template for this category
    template = random.choice(title_templates[category])
    
    # Extract location information
    city = event_row.get('city', 'Local')
    if pd.isna(city):
        city = random.choice(['Local', 'Downtown', 'Regional', 'Community', 'Neighborhood'])
    
    # Get vocabulary for this category
    vocabulary = vocabularies[category]
    
    # Create template variables
    template_vars = {
        'city': city,
        'season': random.choice(['Spring', 'Summer', 'Fall', 'Winter', 'Holiday', 'Weekend']),
        'noun': random.choice(vocabulary['nouns']),
        'adjective': random.choice(vocabulary['adjectives']),
        'verb': random.choice(vocabulary['verbs'])
    }
    
    # Add category-specific variables
    for key in vocabulary:
        if key not in ['nouns', 'adjectives', 'verbs'] and key in template:
            template_vars[key] = random.choice(vocabulary[key])
    
    # Fill the template
    title = template
    for var, value in template_vars.items():
        title = title.replace(f"{{{var}}}", value)
    
    # Capitalize appropriately
    return string.capwords(title)

# Function to generate a description
def generate_description(category, event_row, title, vocabularies, desc_templates):
    """Generate an event description based on category and patterns"""
    # Select a random template
    template = random.choice(desc_templates[category])
    
    # Extract basic event information
    city = event_row.get('city', 'our area')
    if pd.isna(city):
        city = random.choice(['our city', 'the area', 'the region', 'our community', 'the venue'])
    
    # Format time information
    time_info = f"This event takes place on {format_time(event_row.get('start_time'))}."
    
    # Get vocabulary for this category
    vocabulary = vocabularies[category]
    
    # Create template variables
    template_vars = {
        'city': city,
        'time_info': time_info,
        'season': random.choice(['Spring', 'Summer', 'Fall', 'Winter', 'Holiday', 'Weekend']),
        'noun': random.choice(vocabulary['nouns']),
        'adjective': random.choice(vocabulary['adjectives']),
        'verb': random.choice(vocabulary['verbs'])
    }
    
    # Add category-specific variables
    for key in vocabulary:
        if key not in ['nouns', 'adjectives', 'verbs'] and key in template:
            template_vars[key] = random.choice(vocabulary[key])
    
    # Fill the template
    description = template
    for var, value in template_vars.items():
        description = description.replace(f"{{{var}}}", value)
    
    # Add location context and call to action
    state = event_row.get('state', '')
    if not pd.isna(state):
        location = f"{city}, {state}"
    else:
        location = city
    
    description += f" Located in {location}. We hope to see you there!"
    
    return description

# Main function to synthesize event details
def synthesize_event_details(events_df):
    """
    Synthesize event titles, descriptions, and categories without knowing actual word stems
    
    Parameters:
    - events_df: DataFrame containing the events data
    
    Returns:
    - DataFrame with added columns: 'category', 'title', 'description'
    """
    # Get vocabularies and templates
    vocabularies = generate_category_vocabularies()
    title_templates, desc_templates = get_templates()
    
    # Create new columns
    events_df['category'] = ''
    events_df['title'] = ''
    events_df['description'] = ''
    
    # Process each event
    for idx, event in events_df.iterrows():
        # Determine category based on patterns
        category = determine_category(event)
        events_df.at[idx, 'category'] = category
        
        # Generate title
        title = generate_title(category, event, vocabularies, title_templates)
        events_df.at[idx, 'title'] = title
        
        # Generate description
        description = generate_description(category, event, title, vocabularies, desc_templates)
        events_df.at[idx, 'description'] = description
        
        # Progress update for large datasets
        if idx % 1000 == 0:
            print(f"Processed {idx} events")
    
    return events_df

# Function to add variety through slight randomization
def add_variety(events_df):
    """Add variety to avoid repetitive patterns in synthetic data"""
    # Dictionary of alternative phrases for common template parts
    variations = {
        'Join us': ['Come along', 'Be part of', 'Don\'t miss', 'We invite you to', 'Participate in'],
        'experience': ['enjoy', 'discover', 'immerse yourself in', 'be part of', 'engage with'],
        'exciting': ['amazing', 'fantastic', 'wonderful', 'incredible', 'unforgettable'],
        'Perfect for': ['Ideal for', 'Great for', 'Designed for', 'Well-suited for', 'Catering to'],
        'We hope to see you there': ['We look forward to seeing you', 'Join us for this special event', 
                                     'Don\'t miss out', 'Secure your spot today', 'Save the date']
    }
    
    # Apply variations to add variety
    for idx, row in events_df.iterrows():
        desc = row['description']
        for phrase, alternatives in variations.items():
            if phrase in desc and random.random() < 0.7:  # 70% chance of replacing
                desc = desc.replace(phrase, random.choice(alternatives))
        events_df.at[idx, 'description'] = desc
    
    return events_df


