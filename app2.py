import streamlit as st
import pandas as pd
import numpy as np
import json
import openai
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Set your OpenAI API key ----
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---- Streamlit App with Improved UI ----
st.set_page_config(page_title="Maritime Disruption Dataset Generator", layout="wide")
st.title("📊 Maritime Disruption Dataset Generator (1960–Present)")

# ----- Sidebar with Configuration Options -----
with st.sidebar:
    # Create tabs for different configuration categories
    tab1, tab2, tab3, tab4 = st.tabs(["Weights", "Scoring Config", "Data Generation", "Advanced"])
    
    # Tab 1: Weights
    with tab1:
        st.header("Scoring Weights")
        impact_level_weight = st.slider("Impact Level Weight", 0.0, 5.0, 1.5, 0.1)
        chokepoint_exposure_weight = st.slider("Chokepoint Exposure Weight", 0.0, 5.0, 1.0, 0.1)
        past_incidents_weight = st.slider("Past Incidents in Region Weight", 0.0, 5.0, 1.0, 0.1)
        direct_attack_weight = st.slider("Direct Attack Weight", 0.0, 5.0, 1.0, 0.1)
        mitigation_difficulty_weight = st.slider("Mitigation Difficulty Weight", 0.0, 5.0, 1.0, 0.1)
        seasonal_factor_weight = st.slider("Seasonal Factor Weight", 0.0, 5.0, 0.5, 0.1)
    
    # Tab 2: Scoring Configuration
    with tab2:
        st.header("Risk Scoring Configuration")
        
        # Create expandable sections for different scoring components
        with st.expander("Risk Scores", expanded=False):
            risk_threshold = st.slider("Risk Threshold", 1, 10, 8)
            high_risk_score = st.slider("High Risk Score Value", 1, 5, 3)
            
            # Dynamic risk score mapping
            st.subheader("Risk Score Text Mapping")
            high_risk_val = st.number_input("'High' Risk Value", 1, 10, 9)
            medium_risk_val = st.number_input("'Medium' Risk Value", 1, 10, 5)
            low_risk_val = st.number_input("'Low' Risk Value", 1, 10, 2)
        
        with st.expander("Chokepoints", expanded=False):
            st.subheader("Critical Chokepoints")
            default_chokepoints = ["Strait of Hormuz", "Bab el-Mandeb", "Suez Canal", 
                                  "Panama Canal", "Taiwan Strait"]
            chokepoints_text = st.text_area("Enter Critical Chokepoints (one per line)", 
                                           value="\n".join(default_chokepoints))
            critical_chokepoints = [cp.strip() for cp in chokepoints_text.split("\n") if cp.strip()]
            chokepoint_score = st.slider("Chokepoint Score Value", 1, 5, 2)
        
        with st.expander("Regions & Seasons", expanded=False):
            st.subheader("Critical Regions")
            default_regions = ["Middle East", "Adriatic Sea"]
            regions_text = st.text_area("Enter Critical Regions (one per line)", 
                                       value="\n".join(default_regions))
            critical_regions = [r.strip() for r in regions_text.split("\n") if r.strip()]
            region_anomaly_score = st.slider("Region Anomaly Score", 1, 5, 2)
            
            st.subheader("High Impact Seasons")
            season_options = ["Winter", "Spring", "Summer", "Fall"]
            high_impact_seasons = st.multiselect("Select High Impact Seasons", 
                                               options=season_options,
                                               default=["Winter", "Summer"])
            season_score = st.slider("Season Impact Score", 1, 5, 1)
        
        with st.expander("Clusters", expanded=False):
            st.subheader("Critical Clusters")
            default_clusters = ["Long-Term Geopolitical Conflicts", "Acute Chokepoint or Port Disruptions"]
            clusters_text = st.text_area("Enter Critical Clusters (one per line)", 
                                        value="\n".join(default_clusters))
            critical_clusters = [c.strip() for c in clusters_text.split("\n") if c.strip()]
            cluster_anomaly_score = st.slider("Cluster Anomaly Score", 1, 5, 2)
        
        with st.expander("Impact & Response", expanded=False):
            st.subheader("Impact Level Mapping")
            high_impact_val = st.number_input("'High' Impact Value", 1, 5, 3)
            medium_impact_val = st.number_input("'Medium' Impact Value", 1, 5, 2)
            low_impact_val = st.number_input("'Low' Impact Value", 1, 5, 1)
            
            st.subheader("Response Mechanism Mapping")
            proactive_val = st.number_input("'Proactive' Response Value", 1, 5, 1)
            reactive_val = st.number_input("'Reactive' Response Value", 1, 5, 2)
            adaptive_val = st.number_input("'Adaptive/International' Response Value", 1, 5, 3)
        
        with st.expander("Warning Thresholds", expanded=False):
            st.subheader("Warning Level Thresholds")
            high_warning_threshold = st.slider("High Warning Threshold", 1, 20, 7)
            medium_warning_threshold = st.slider("Medium Warning Threshold", 1, 20, 4)
    
    # Tab 3: Data Generation Settings
    with tab3:
        st.header("Dataset Size")
        dataset_size = st.slider("Number of events to generate", 10, 500, 50, 10)
        
        # Default start/end years with full range
        start_year = st.number_input("Start Year", 1960, 2024, 1960)
        end_year = st.number_input("End Year", 1960, 2024, 2024)
        
        st.header("Year Distribution")
        year_distribution = st.selectbox(
            "Distribution of events across years",
            ["Even across all decades", "More recent events", "Random distribution"],
            index=0
        )
        
        st.header("Cost Reduction Options")
        st.info("💡 Smaller values = lower API costs")
        use_seed = st.checkbox("Use Seed for Consistency", value=True)
        consistency_seed = st.number_input("Consistency Seed", 1, 99999, 12345) if use_seed else None
        
        # Set a higher default for better coverage
        max_api_calls = st.slider("Maximum API Calls", 1, 20, 7, help="Higher = better decade coverage, but more expensive")
        
    # Tab 4: Advanced Settings
    with tab4:
        st.header("Performance Options")
        parallel_workers = st.slider("Parallel API calls", 1, 5, 3)
        batch_size = st.slider("Events per API call", 5, 40, 20, help="Higher = fewer API calls but might time out")
        chunk_size = st.slider("Processing chunk size", 100, 2000, 500)
        
        st.header("API Settings")
        model_name = st.selectbox("OpenAI Model", 
                                 ["gpt-4o-mini"], 
                                 index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)

# Compile weights and scoring configuration
weights = {
    'impact_level': impact_level_weight,
    'chokepoint_exposure': chokepoint_exposure_weight,
    'past_incidents': past_incidents_weight,
    'direct_attack': direct_attack_weight,
    'mitigation_difficulty': mitigation_difficulty_weight,
    'seasonal_factor': seasonal_factor_weight
}

# Compile the scoring configuration from all inputs
scoring_config = {
    # Risk scores
    'risk_threshold': risk_threshold,
    'high_risk_score': high_risk_score,
    'risk_score_map': {"High": high_risk_val, "Medium": medium_risk_val, "Low": low_risk_val},
    
    # Chokepoints
    'critical_chokepoints': critical_chokepoints,
    'chokepoint_score': chokepoint_score,
    
    # Regions and Seasons
    'critical_regions': critical_regions,
    'region_anomaly_score': region_anomaly_score,
    'high_impact_seasons': high_impact_seasons,
    'season_score': season_score,
    
    # Clusters
    'critical_clusters': critical_clusters,
    'cluster_anomaly_score': cluster_anomaly_score,
    
    # Impact and Response
    'impact_level_map': {"High": high_impact_val, "Medium": medium_impact_val, "Low": low_impact_val},
    'response_mechanism_map': {
        "proactive": proactive_val,
        "reactive": reactive_val, 
        "adaptive": adaptive_val,
        "international": adaptive_val
    },
    
    # Warning thresholds
    'high_warning_threshold': high_warning_threshold,
    'medium_warning_threshold': medium_warning_threshold
}

# ---- Calculate scores with dynamic configuration ----
def calculate_scores(chunk_df, weights, scoring_config):
    """
    Optimized vectorized calculation of scores for a chunk of the dataframe
    with dynamic scoring configuration
    """
    # Initialize score columns
    for col in ['Score_Risk', 'Score_Chokepoint', 'Score_Season', 'Score_RegionAnomaly', 'Score_ClusterAnomaly']:
        chunk_df[col] = 0
    
    # Convert Risk Score from string to number if needed
    risk_scores = chunk_df.get('Risk Score', pd.Series([0] * len(chunk_df)))
    if risk_scores.dtype == 'object':
        # Use user-defined risk score mapping
        risk_score_map = scoring_config['risk_score_map']
        numeric_risk_scores = risk_scores.map(lambda x: risk_score_map.get(x, 0) if isinstance(x, str) else x)
    else:
        numeric_risk_scores = risk_scores
    
    # Calculate Score_Risk using dynamic threshold
    chunk_df['Score_Risk'] = np.where(numeric_risk_scores >= scoring_config['risk_threshold'], 
                                      scoring_config['high_risk_score'], 0)
    
    # Calculate Score_Chokepoint
    chokepoints = chunk_df.get('Chokepoint (if any)', pd.Series([''] * len(chunk_df)))
    if chokepoints.isna().all() and 'Chokepoint' in chunk_df.columns:
        chokepoints = chunk_df['Chokepoint']
    
    # Use user-defined critical chokepoints
    critical_chokepoints = scoring_config['critical_chokepoints']
    chunk_df['Score_Chokepoint'] = np.where(
        chokepoints.isin(critical_chokepoints), scoring_config['chokepoint_score'], 0
    )
    
    # Calculate Score_Season
    seasons = chunk_df.get('Season', pd.Series([''] * len(chunk_df)))
    chunk_df['Score_Season'] = np.where(
        seasons.isin(scoring_config['high_impact_seasons']), 
        scoring_config['season_score'], 0
    )
    
    # Calculate Score_RegionAnomaly
    regions = chunk_df.get('Region', pd.Series([''] * len(chunk_df)))
    chunk_df['Score_RegionAnomaly'] = np.where(
        regions.isin(scoring_config['critical_regions']), 
        scoring_config['region_anomaly_score'], 0
    )
    
    # Calculate Score_ClusterAnomaly
    clusters = chunk_df.get('Cluster Label', pd.Series([''] * len(chunk_df)))
    chunk_df['Score_ClusterAnomaly'] = np.where(
        clusters.isin(scoring_config['critical_clusters']), 
        scoring_config['cluster_anomaly_score'], 0
    )
    
    # Handle Impact Level string values
    impact_levels = chunk_df.get('Impact Level', pd.Series([0] * len(chunk_df)))
    if impact_levels.dtype == 'object':
        impact_level_map = scoring_config['impact_level_map']
        numeric_impacts = impact_levels.map(lambda x: impact_level_map.get(x, 0) if isinstance(x, str) else x)
    else:
        numeric_impacts = impact_levels
    
    # Handle Response Mechanism with user-defined mapping
    response_mechs = chunk_df.get('Response Mechanism', pd.Series([0] * len(chunk_df)))
    if response_mechs.dtype == 'object':
        response_map = scoring_config['response_mechanism_map']
        
        def map_response(x):
            if not isinstance(x, str):
                return 0
            x_lower = x.lower()
            for key, value in response_map.items():
                if key.lower() in x_lower:
                    return value
            return 0
        
        response_values = response_mechs.map(map_response)
    else:
        response_values = response_mechs
    
    # Calculate final scores using user-defined weights
    chunk_df['Warning Score'] = (
        weights['impact_level'] * numeric_impacts +
        weights['chokepoint_exposure'] * chunk_df['Score_Chokepoint'] +
        weights['past_incidents'] * (chunk_df['Score_RegionAnomaly'] + chunk_df['Score_ClusterAnomaly']) +
        weights['direct_attack'] * numeric_risk_scores +
        weights['mitigation_difficulty'] * (response_values + chunk_df['Score_Chokepoint']) +
        weights['seasonal_factor'] * chunk_df['Score_Season']
    )
    
    # Set Warning Level based on dynamic Warning Score thresholds
    conditions = [
        chunk_df['Warning Score'] >= scoring_config['high_warning_threshold'],
        chunk_df['Warning Score'] >= scoring_config['medium_warning_threshold']
    ]
    choices = ['High', 'Medium']
    chunk_df['Warning Level'] = np.select(conditions, choices, default='Low')
    
    return chunk_df

# ---- Extract JSON from response with improved error handling ----
def extract_json_from_response(text):
    """Extract JSON array from text that might contain extra content"""
    try:
        # Try direct parsing first (fastest)
        return json.loads(text)
    except:
        pass
    
    # Find JSON array pattern
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    # Remove markdown code blocks if present
    cleaned_text = re.sub(r'```json\s+|\s+```', '', text)
    cleaned_text = re.sub(r'```\s+|\s+```', '', cleaned_text)
    
    # Try to find JSON array again
    json_match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    # Try to find JSON object and extract array
    json_obj_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
    if json_obj_match:
        try:
            json_obj = json.loads(json_obj_match.group(0))
            # Look for array fields
            for key, value in json_obj.items():
                if isinstance(value, list):
                    return value
        except:
            pass
    
    # If all else fails
    raise ValueError("Could not extract valid JSON from the response")

# ---- Generate batch of events with consistency and required fields ----
def generate_events_batch(batch_size, base_year, end_year, seed=None, retries=2):
    """Generate a batch of maritime events using OpenAI API with retry logic and consistency"""
    
    # Add seed for consistency if provided
    seed_text = f" Use seed {seed} for consistency." if seed else ""
    
    # Comprehensive prompt with all required fields and strong instruction for year range
    prompt = f"""
    Generate a JSON array of {batch_size} maritime disruption events from {base_year} to {end_year}.{seed_text}
    IMPORTANT: Events MUST be EVENLY distributed across the entire time period from {base_year} to {end_year}.
    DO NOT focus only on recent years or early years - distribute events across ALL decades.
    
    Include ALL of these EXACT field names in valid JSON format:
    - Event: brief description
    - Year(s): when it happened (MUST be between {base_year} and {end_year})
    - Region: maritime region affected
    - Affected Segment(s): part of maritime industry affected
    - Impact Summary: brief summary of impact
    - Event Category: category of the event
    - Impact Level: High, Medium, or Low
    - Affected Commodity: main commodity affected
    - Response Mechanism: Proactive, Reactive, or Adaptive
    - Response Type: specific type of response
    - Start Month: month event started
    - End Month: month event ended
    - Estimated Duration: duration in months
    - Risk Type: type of risk posed
    - Insurance Impact: impact on insurance
    - Trade Routes Affected: main trade routes affected
    - Positive Outcome / Opportunity: any positive outcomes
    - Source / Reference Link: leave blank or use placeholder
    - AI Event Summary: AI-generated summary
    - RAG_Context: contextual information
    - Chokepoint (if any): relevant shipping chokepoint
    - Cluster: general cluster category
    - Duration_Months: numerical duration in months
    - Cluster Label: more specific event category
    - Start Year: year event started (MUST be between {base_year} and {end_year})
    - Decade: which decade it occurred in (e.g. "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s")
    - Season: Winter, Spring, Summer, or Fall
    - Trend Flag: Yes/No if part of a trend
    - Risk Score: 1-10 scale
    - lat: latitude coordinate (approximate)
    - lon: longitude coordinate (approximate)
    - Anomaly_ZScore: placeholder value between -3 and 3
    - Anomaly_IsoForest: placeholder value between 0 and 1
    - Anomaly_Detected: Yes/No if anomaly detected

    Return ONLY a valid JSON array with no additional text or explanations.
    """
    
    for attempt in range(retries):
        try:
            # Use a system message that enforces JSON-only response
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a maritime data generator. Return ONLY valid JSON with no additional text. Make sure ALL required fields are included for EVERY event. DISTRIBUTE EVENTS EVENLY ACROSS ALL DECADES FROM 1960 to 2024."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=4000
            )
            
            # Get the content
            raw_content = response.choices[0].message.content.strip()
            
            # Try to parse the JSON
            try:
                events = extract_json_from_response(raw_content)
                
                # Validate that all events have the required fields
                required_fields = ['Event', 'Year(s)', 'Region', 'Affected Segment(s)', 
                                  'Impact Summary', 'Event Category', 'Impact Level',
                                  'Affected Commodity', 'Response Mechanism', 'Response Type',
                                  'Start Month', 'End Month', 'Estimated Duration', 'Risk Type',
                                  'Insurance Impact', 'Trade Routes Affected', 
                                  'Positive Outcome / Opportunity', 'Source / Reference Link',
                                  'AI Event Summary', 'RAG_Context', 'Chokepoint (if any)', 
                                  'Cluster', 'Duration_Months', 'Cluster Label', 'Start Year',
                                  'Decade', 'Season', 'Trend Flag', 'Risk Score', 
                                  'lat', 'lon', 'Anomaly_ZScore', 'Anomaly_IsoForest', 'Anomaly_Detected']
                
                # Check for missing fields
                valid_events = []
                for event in events:
                    missing_fields = [field for field in required_fields if field not in event]
                    if not missing_fields:
                        valid_events.append(event)
                    else:
                        # Add default values for missing fields
                        for field in missing_fields:
                            # Set default values based on field type
                            if field == 'Impact Level':
                                event[field] = 'Medium'
                            elif field == 'Response Mechanism':
                                event[field] = 'Reactive'
                            elif field == 'Chokepoint (if any)':
                                event[field] = 'None'
                            elif field == 'Season':
                                event[field] = 'Summer'
                            elif field == 'Risk Score':
                                event[field] = 5
                            elif field == 'lat' or field == 'lon':
                                event[field] = 0.0
                            elif field == 'Start Year':
                                try:
                                    year_str = str(event.get('Year(s)', ''))
                                    if '-' in year_str:
                                        event[field] = int(year_str.split('-')[0])
                                    else:
                                        event[field] = int(year_str) if year_str.isdigit() else base_year
                                except:
                                    event[field] = base_year
                            elif field == 'Decade':
                                try:
                                    start_yr = event.get('Start Year', 0)
                                    if start_yr == 0 and 'Year(s)' in event:
                                        year_str = str(event['Year(s)'])
                                        if '-' in year_str:
                                            start_yr = int(year_str.split('-')[0])
                                        else:
                                            start_yr = int(year_str) if year_str.isdigit() else base_year
                                    
                                    decade = (start_yr // 10) * 10
                                    event[field] = f"{decade}s"
                                except:
                                    event[field] = f"{(base_year // 10) * 10}s"
                            elif field == 'Trend Flag':
                                event[field] = 'No'
                            elif field == 'Anomaly_ZScore':
                                event[field] = 0.0
                            elif field == 'Anomaly_IsoForest':
                                event[field] = 0.5
                            elif field == 'Anomaly_Detected':
                                event[field] = 'No'
                            elif field == 'Duration_Months':
                                event[field] = 1
                            elif field == 'Start Month' or field == 'End Month':
                                event[field] = 'January'
                            elif field == 'Estimated Duration':
                                event[field] = '1 month'
                            else:
                                event[field] = 'Unknown'
                        valid_events.append(event)
                
                # Validate years are in the correct range and fix if necessary
                for event in valid_events:
                    # Fix Start Year if needed
                    try:
                        start_year_val = int(event['Start Year'])
                        if start_year_val < base_year or start_year_val > end_year:
                            # Assign a random year in the range
                            event['Start Year'] = np.random.randint(base_year, end_year + 1)
                    except:
                        event['Start Year'] = np.random.randint(base_year, end_year + 1)
                    
                    # Update Decade based on Start Year
                    decade = (int(event['Start Year']) // 10) * 10
                    event['Decade'] = f"{decade}s"
                    
                    # Update Year(s) for consistency
                    event['Year(s)'] = str(event['Start Year'])
                
                return valid_events
                
            except Exception as json_error:
                if attempt < retries - 1:
                    time.sleep(1)  # Short wait before retry
                    continue
                else:
                    raise json_error
                
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)  # Short wait before retry
                continue
            else:
                st.warning(f"API error: {str(e)}")
                # Return partial data to avoid complete failure
                return []
    
    return []

# ---- Create decade ranges for more comprehensive coverage ----
def create_decade_ranges(start_year, end_year, distribution_type="even"):
    """Create decade ranges based on the selected distribution type"""
    # Define all possible decades
    all_decades = []
    for decade_start in range((start_year // 10) * 10, (end_year // 10) * 10 + 1, 10):
        decade_end = min(decade_start + 9, end_year)
        if decade_end >= start_year:  # Only include if part of the decade is in our range
            all_decades.append((decade_start, decade_end))
    
    if distribution_type == "even":
        # Return all decades with equal weight
        return all_decades
    
    elif distribution_type == "more_recent":
        # Weight decades by recency
        weighted_decades = []
        for i, decade in enumerate(all_decades):
            # Add each decade multiple times based on its position
            weight = i + 1  # 1 for earliest decade, increasing for more recent ones
            weighted_decades.extend([decade] * weight)
        return weighted_decades
    
    else:  # Random distribution
        # Return all decades (random selection will happen later)
        return all_decades

# ---- Cost-optimized event generation with improved decade distribution ----
def generate_events_with_limits(total_size, workers=2, batch_size=20, year_range=(1960, 2024), 
                               distribution="even", consistency_seed=None, max_api_calls=7):
    """Generate events with improved decade distribution and API call limits"""
    all_events = []
    start_year, end_year = year_range
    
    # Calculate optimal batch distribution
    events_per_api_call = min(batch_size * workers, total_size)
    api_calls_needed = (total_size + events_per_api_call - 1) // events_per_api_call
    api_calls_to_make = min(api_calls_needed, max_api_calls)
    
    # Calculate total events we can generate with the API calls limit
    events_to_generate = min(total_size, api_calls_to_make * events_per_api_call)
    
    if api_calls_to_make < api_calls_needed:
        st.warning(f"⚠️ Limiting to {events_to_generate} events to stay within {max_api_calls} API calls. Adjust 'Maximum API Calls' in settings if you need more events.")
    
    # Create decade ranges based on selected distribution
    if distribution == "Even across all decades":
        dist_type = "even"
    elif distribution == "More recent events":
        dist_type = "more_recent"
    else:
        dist_type = "random"
    
    # Get decade ranges
    all_decades = create_decade_ranges(start_year, end_year, dist_type)
    
    # Ensure we have at least one range
    if not all_decades:
        all_decades = [(start_year, end_year)]
    
    # Create batches to ensure coverage of all decades
    batches = []
    events_per_decade = events_to_generate // len(all_decades)
    extra_events = events_to_generate % len(all_decades)
    
    # First distribute events_per_decade to each decade
    for decade in all_decades:
        remaining = events_per_decade
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            batches.append((current_batch, decade))
            remaining -= current_batch
    
    # Then distribute any extra events
    decade_idx = 0
    for _ in range(extra_events):
        # Find the first batch for the current decade and add 1
        for i, batch in enumerate(batches):
            if batch[1] == all_decades[decade_idx]:
                new_size = batch[0] + 1
                batches[i] = (new_size, batch[1])
                break
        decade_idx = (decade_idx + 1) % len(all_decades)
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Submit API calls in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Use unique seeds derived from the main seed for consistency
        future_to_batch = {
            executor.submit(
                generate_events_batch, 
                batch[0], 
                batch[1][0],  # decade start
                batch[1][1],  # decade end
                None if consistency_seed is None else consistency_seed + i
            ): i for i, batch in enumerate(batches)
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_batch):
            try:
                batch_events = future.result()
                all_events.extend(batch_events)
                completed += 1
                progress = completed / len(batches)
                progress_bar.progress(progress)
                status_text.text(f"Generated {len(all_events)} events out of {events_to_generate} ({int(progress*100)}%)")
            except Exception as e:
                st.warning(f"⚠️ Batch generation error: {str(e)}")
    
    return all_events

# ---- Process data in chunks ----
def process_dataframe_in_chunks(df, weights, scoring_config, chunk_size=500):
    """Process a large dataframe in chunks to avoid memory issues"""
    result_dfs = []
    
    # Split dataframe into chunks
    chunk_progress = st.progress(0)
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        # Process each chunk
        processed_chunk = calculate_scores(chunk, weights, scoring_config)
        result_dfs.append(processed_chunk)
        
        # Update progress
        chunk_progress.progress((i + 1) / len(chunks))
    
    # Combine results
    return pd.concat(result_dfs, ignore_index=True)

# --- Generate Dataset ---
col1, col2 = st.columns([2, 1])
with col1:
    generate_button = st.button("🚀 Generate Dataset", type="primary", use_container_width=True)

with col2:
    if st.button("📋 View Configuration", use_container_width=True):
        st.json({
            "weights": weights,
            "scoring_config": scoring_config,
            "generation_settings": {
                "dataset_size": dataset_size,
                "year_range": [start_year, end_year],
                "year_distribution": year_distribution,
                "consistency_seed": consistency_seed if use_seed else "Not used",
                "max_api_calls": max_api_calls,
                "parallel_workers": parallel_workers,
                "batch_size": batch_size,
                "model": model_name
            }
        })

if generate_button:
    try:
        # Generate events with improved decade distribution
        st.subheader("Step 1: Generating Events")
        
        # Use dynamic year distribution
        dist_type_map = {
            "Even across all decades": "even",
            "More recent events": "more_recent",
            "Random distribution": "random"
        }
        
        # Tell user what we're doing
        st.info(f"🔄 Generating events from {start_year} to {end_year} with {year_distribution} distribution")
        st.info(f"🔄 Using maximum {max_api_calls} API calls to balance coverage and costs")
        
        events = generate_events_with_limits(
            total_size=dataset_size,
            workers=parallel_workers,
            batch_size=batch_size,
            year_range=(start_year, end_year),
            distribution=year_distribution,
            consistency_seed=consistency_seed if use_seed else None,
            max_api_calls=max_api_calls
        )
        
        if not events:
            st.error("Failed to generate any events. Please try again or adjust settings.")
        else:
            # Convert to DataFrame
            df = pd.DataFrame(events)
            
            # Ensure all required columns exist with appropriate default values
            required_columns = ['Event', 'Year(s)', 'Region', 'Affected Segment(s)', 
                              'Impact Summary', 'Event Category', 'Impact Level',
                              'Affected Commodity', 'Response Mechanism', 'Response Type',
                              'Start Month', 'End Month', 'Estimated Duration', 'Risk Type',
                              'Insurance Impact', 'Trade Routes Affected', 
                              'Positive Outcome / Opportunity', 'Source / Reference Link',
                              'AI Event Summary', 'RAG_Context', 'Chokepoint (if any)', 
                              'Cluster', 'Duration_Months', 'Cluster Label', 'Start Year',
                              'Decade', 'Season', 'Trend Flag', 'Risk Score', 
                              'lat', 'lon', 'Anomaly_ZScore', 'Anomaly_IsoForest', 'Anomaly_Detected']
                               
            for col in required_columns:
                if col not in df.columns:
                    # Add missing columns with appropriate default values
                    if col == 'Impact Level':
                        df[col] = 'Medium'
                    elif col == 'Response Mechanism':
                        df[col] = 'Reactive'
                    elif col == 'Chokepoint (if any)':
                        df[col] = 'None'
                    elif col == 'Season':
                        df[col] = 'Summer'
                    elif col == 'Risk Score':
                        df[col] = 5
                    elif col == 'lat' or col == 'lon':
                        df[col] = 0.0
                    elif col == 'Decade':
                        # Calculate decade based on Start Year
                        if 'Start Year' in df.columns:
                            df[col] = df['Start Year'].apply(lambda x: f"{(int(x) // 10) * 10}s")
                        else:
                            decades = [f"{d}0s" for d in range(196, 203)]
                            df[col] = df['Year(s)'].apply(lambda x: next((d for d in decades if str(d)[0:3] in str(x)), "2020s"))
                    elif col == 'Trend Flag':
                        df[col] = 'No'
                    elif col == 'Anomaly_ZScore':
                        df[col] = np.random.uniform(-2, 2, len(df))
                    elif col == 'Anomaly_IsoForest':
                        df[col] = np.random.uniform(0, 1, len(df))
                    elif col == 'Anomaly_Detected':
                        df[col] = np.random.choice(['Yes', 'No'], len(df), p=[0.2, 0.8])
                    elif col == 'Duration_Months':
                        df[col] = np.random.randint(1, 24, len(df))
                    elif col == 'Start Month':
                        months = ['January', 'February', 'March', 'April', 'May', 'June', 
                                 'July', 'August', 'September', 'October', 'November', 'December']
                        df[col] = np.random.choice(months, len(df))
                    elif col == 'End Month':
                        months = ['January', 'February', 'March', 'April', 'May', 'June', 
                                 'July', 'August', 'September', 'October', 'November', 'December']
                        df[col] = np.random.choice(months, len(df))
                    elif col == 'Start Year':
                        # Extract the start year from Year(s)
                        df[col] = df['Year(s)'].apply(lambda x: int(str(x).split('-')[0]) if '-' in str(x) else int(str(x)))
                    else:
                        df[col] = 'Unknown'
            
            # Run a validation to ensure all decade fields are correct
            df['Decade'] = df['Start Year'].apply(lambda x: f"{(int(x) // 10) * 10}s")
            
            st.success(f"✅ Generated {len(df)} events successfully!")
            
            # Show decade distribution
            decade_counts = df['Decade'].value_counts().sort_index()
            st.subheader("Events by Decade")
            st.bar_chart(decade_counts)
            
            # Process in chunks
            st.subheader("Step 2: Calculating Risk Scores")
            df = process_dataframe_in_chunks(df, weights, scoring_config, chunk_size)
            
            # Store in session state
            st.session_state['maritime_events_df'] = df
            
            # Display preview
            st.subheader("Dataset Preview")
            st.dataframe(df, use_container_width=True)
            
            # Statistics
            stats_cols = st.columns(3)
            with stats_cols[0]:
                st.metric("Total Events", len(df))
            with stats_cols[1]:
                high_risk = len(df[df['Warning Level'] == 'High'])
                st.metric("High Risk Events", high_risk, f"{high_risk/len(df):.1%}")
            with stats_cols[2]:
                regions_count = df['Region'].nunique() if 'Region' in df.columns else 0
                st.metric("Unique Regions", regions_count)
            
            # Download options
            st.subheader("Download Options")
            download_cols = st.columns(2)
            
            with download_cols[0]:
                # Excel download with complete data
                excel_io = io.BytesIO()
                with pd.ExcelWriter(excel_io, engine='openpyxl') as writer:
                    # Write the main data
                    df.to_excel(writer, index=False, sheet_name="Maritime Events")
                    
                    # Write configuration to a separate sheet
                    config_df = pd.DataFrame({
                        "Parameter": ["Weights", "Scoring Configuration", "Dataset Size", "Year Range", "Distribution"],
                        "Value": [
                            str(weights),
                            str(scoring_config),
                            str(dataset_size),
                            f"{start_year} to {end_year}",
                            year_distribution
                        ]
                    })
                    config_df.to_excel(writer, index=False, sheet_name="Configuration")
                    
                    # Add summary statistics
                    warning_counts = df['Warning Level'].value_counts().reset_index()
                    warning_counts.columns = ['Warning Level', 'Count']
                    warning_counts.to_excel(writer, index=False, sheet_name="Summary")
                    
                    # Add decade distribution
                    decade_df = df['Decade'].value_counts().sort_index().reset_index()
                    decade_df.columns = ['Decade', 'Count']
                    decade_df.to_excel(writer, index=False, sheet_name="Decade Distribution")
                    
                excel_io.seek(0)
                
                st.download_button(
                    "📥 Download Complete Excel",
                    data=excel_io,
                    file_name="maritime_events_complete.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with download_cols[1]:
                # CSV download
                csv_io = io.BytesIO()
                df.to_csv(csv_io, index=False)
                csv_io.seek(0)
                
                st.download_button(
                    "📥 Download CSV",
                    data=csv_io,
                    file_name="maritime_events.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"Error generating dataset: {str(e)}")
        st.error("Try adjusting the parameters or running again.")

# Add enhanced visualization if data exists
if 'maritime_events_df' in st.session_state:
    st.subheader("Data Visualization")
    viz_df = st.session_state['maritime_events_df']
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Warning Levels", "Regions", "Timeline", "Decades"])
    
    # Tab 1: Warning Level Distribution
    with viz_tab1:
        if 'Warning Level' in viz_df.columns:
            st.write("Warning Level Distribution:")
            warning_counts = viz_df['Warning Level'].value_counts()
            st.bar_chart(warning_counts)
    
    # Tab 2: Region Analysis
    with viz_tab2:
        if 'Region' in viz_df.columns:
            st.write("Events by Region:")
            region_counts = viz_df['Region'].value_counts().head(10)
            st.bar_chart(region_counts)
    
    # Tab 3: Timeline Analysis
    with viz_tab3:
        if 'Start Year' in viz_df.columns:
            st.write("Events by Year:")
            year_counts = viz_df['Start Year'].value_counts().sort_index()
            st.line_chart(year_counts)
    
    # Tab 4: Decade Analysis
    with viz_tab4:
        if 'Decade' in viz_df.columns:
            st.write("Events by Decade:")
            decade_counts = viz_df['Decade'].value_counts().sort_index()
            st.bar_chart(decade_counts)