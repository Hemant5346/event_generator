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
        
        st.header("Cost Reduction Options")
        st.info("💡 Adjust these settings to balance cost vs. dataset size")
        use_seed = st.checkbox("Use Seed for Consistency", value=True)
        consistency_seed = st.number_input("Consistency Seed", 1, 99999, 12345) if use_seed else None
        max_api_calls = st.slider("Maximum API Calls", 1, 20, 5, help="Higher = more data but more expensive")
        
        with st.expander("Advanced Generation Settings", expanded=False):
            st.warning("If you're not getting enough events, increase these values:")
            min_events_per_batch = st.slider("Minimum Events Per Batch", 5, 50, 25, 
                                           help="Higher = more events per API call but may cause timeouts")
            st.info(f"With current settings, you can generate approximately {min_events_per_batch * max_api_calls} events")
        
    # Tab 4: Advanced Settings
    with tab4:
        st.header("Performance Options")
        parallel_workers = st.slider("Parallel API calls", 1, 4, 2)
        batch_size = st.slider("Events per API call", 10, 50, 30, help="Higher = fewer API calls but might time out")
        chunk_size = st.slider("Processing chunk size", 100, 2000, 500)
        
        with st.expander("Batch Size Calculator", expanded=False):
            st.info("This will help you determine the optimal batch size to get all your events")
            target_events = st.number_input("Target number of events", 10, 500, dataset_size)
            available_api_calls = st.number_input("Available API calls", 1, 20, max_api_calls)
            workers_count = st.number_input("Parallel workers", 1, 4, parallel_workers)
            
            min_batch_size = (target_events + (available_api_calls * workers_count) - 1) // (available_api_calls * workers_count)
            st.success(f"To generate {target_events} events with {available_api_calls} API calls and {workers_count} workers, use a batch size of at least {min_batch_size}")
            
            if min_batch_size > 50:
                st.warning(f"⚠️ A batch size of {min_batch_size} might cause timeouts. Consider increasing API calls or reducing target events.")
        
        st.header("API Settings")
        model_name = st.selectbox("OpenAI Model", 
                                 ["gpt-3.5-turbo"], 
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
    if 'Chokepoint' in chunk_df.columns and chokepoints.isna().all():
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
    
    # Calculate Trigger Reason - a text description of why this event has its warning level
    def determine_trigger_reason(row):
        reasons = []
        if row['Score_Risk'] > 0:
            reasons.append("High Risk Score")
        if row['Score_Chokepoint'] > 0:
            reasons.append("Critical Chokepoint")
        if row['Score_RegionAnomaly'] > 0:
            reasons.append("Critical Region")
        if row['Score_ClusterAnomaly'] > 0:
            reasons.append("Critical Cluster")
        if row['Score_Season'] > 0:
            reasons.append("High Impact Season")
        
        if not reasons:
            return "Combined Factors"
        elif len(reasons) == 1:
            return reasons[0]
        else:
            return " & ".join(reasons[:2])
    
    # Apply the function to create the Trigger Reason column
    chunk_df['Trigger Reason'] = chunk_df.apply(determine_trigger_reason, axis=1)
    
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

# ---- Generate batch of events with decade focus ----
def generate_events_batch(batch_size, base_year, end_year, seed=None, retries=2):
    """Generate a batch of maritime events for a specific decade using OpenAI API with retry logic"""
    
    # Add seed for consistency if provided
    seed_text = f" Use seed {seed} for consistency." if seed else ""
    
    # Decade-focused prompt to ensure we get events from the specified timeframe
    decade_name = f"{base_year}s"
    
    prompt = f"""
    Generate a JSON array of {batch_size} maritime disruption events SPECIFICALLY from the {decade_name} (from {base_year} to {end_year} only).{seed_text}
    Do NOT include events from other decades - ONLY events from {base_year} to {end_year}.
    
    Include these fields in valid JSON format:
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
    - Decade: "{decade_name}"
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
            # Use a system message that enforces JSON-only response with shortened prompt
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a maritime data generator. Return ONLY valid JSON with no additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=4000   # Limit token usage
            )
            
            # Get the content
            raw_content = response.choices[0].message.content.strip()
            
            # Try to parse the JSON
            try:
                return extract_json_from_response(raw_content)
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

# ---- Decade-focused event generation ----
def generate_events_with_decades(total_size, workers=2, batch_size=20, consistency_seed=None, max_api_calls=3):
    """Generate events with focus on covering all decades equally"""
    all_events = []
    
    # Create decade ranges 
    decades = [
        (1960, 1969, "1960s"), 
        (1970, 1979, "1970s"), 
        (1980, 1989, "1980s"), 
        (1990, 1999, "1990s"),
        (2000, 2009, "2000s"), 
        (2010, 2019, "2010s"), 
        (2020, 2024, "2020s")
    ]
    
    # Calculate events per decade - try to distribute evenly
    num_decades = len(decades)
    events_per_decade = total_size // num_decades
    
    # Ensure at least some events per decade
    events_per_decade = max(events_per_decade, 5)
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Generate events decade by decade to ensure coverage
    decade_batches = []
    
    # If we have very limited API calls, we need to combine decades
    if max_api_calls < num_decades:
        # Group decades to fit within API call limit
        decades_per_call = (num_decades + max_api_calls - 1) // max_api_calls
        
        # Create combined decade batches
        for i in range(0, len(decades), decades_per_call):
            combined_decades = decades[i:i+decades_per_call]
            if combined_decades:
                start_year = combined_decades[0][0]
                end_year = combined_decades[-1][1]
                decade_label = f"{combined_decades[0][2]}-{combined_decades[-1][2]}"
                decade_batches.append((start_year, end_year, decade_label, events_per_decade * len(combined_decades)))
    else:
        # We have enough API calls for each decade
        for start_year, end_year, decade_label in decades:
            decade_batches.append((start_year, end_year, decade_label, events_per_decade))
    
    # Submit API calls for each decade batch
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Track progress
        total_batches = len(decade_batches)
        completed_batches = 0
        
        # Process each decade batch
        for decade_idx, (start_year, end_year, decade_label, target_events) in enumerate(decade_batches):
            try:
                status_text.text(f"Generating events for {decade_label}...")
                
                # Calculate API calls needed for this decade batch
                api_calls_for_decade = min(
                    (target_events + batch_size - 1) // batch_size,  # Ceiling division
                    max_api_calls // total_batches + (1 if decade_idx < max_api_calls % total_batches else 0)  # Fair distribution
                )
                
                # Adjust batch size if needed
                adjusted_batch_size = min(batch_size, (target_events + api_calls_for_decade - 1) // api_calls_for_decade)
                
                # Create sub-batches for this decade
                sub_batches = []
                remaining = target_events
                for i in range(api_calls_for_decade):
                    if remaining <= 0:
                        break
                    sub_batch_size = min(adjusted_batch_size, remaining)
                    sub_batches.append(sub_batch_size)
                    remaining -= sub_batch_size
                
                # Generate events for each sub-batch
                decade_futures = []
                for i, sub_batch_size in enumerate(sub_batches):
                    if sub_batch_size > 0:
                        # Create a unique seed for consistency if needed
                        batch_seed = None if consistency_seed is None else consistency_seed + decade_idx * 1000 + i
                        decade_futures.append(
                            executor.submit(
                                generate_events_batch,
                                sub_batch_size,
                                start_year,
                                end_year,
                                batch_seed
                            )
                        )
                
                # Collect results for this decade
                decade_events = []
                for future in as_completed(decade_futures):
                    try:
                        batch_events = future.result()
                        # Ensure events are actually from this decade
                        filtered_events = []
                        for event in batch_events:
                            # Try to get the year from different possible fields
                            year = None
                            if 'Start Year' in event and event['Start Year']:
                                try:
                                    year = int(event['Start Year'])
                                except:
                                    pass
                            
                            if year is None and 'Year(s)' in event and event['Year(s)']:
                                try:
                                    # Try to extract a year from the Year(s) field
                                    year_str = str(event['Year(s)'])
                                    year_match = re.search(r'\b(19\d\d|20[0-2]\d)\b', year_str)
                                    if year_match:
                                        year = int(year_match.group(1))
                                except:
                                    pass
                            
                            # Only add events that match the decade
                            if year is not None and start_year <= year <= end_year:
                                # Ensure the Decade field is set correctly
                                event['Decade'] = decade_label
                                filtered_events.append(event)
                        
                        decade_events.extend(filtered_events)
                    except Exception as e:
                        st.warning(f"⚠️ Error generating events for {decade_label}: {str(e)}")
                
                all_events.extend(decade_events)
                
                # Update progress
                completed_batches += 1
                progress = completed_batches / total_batches
                progress_bar.progress(progress)
                status_text.text(f"Generated {len(decade_events)} events for {decade_label} ({int(progress*100)}% complete)")
                
            except Exception as e:
                st.warning(f"⚠️ Error processing {decade_label}: {str(e)}")
    
    # Final status update
    status_text.text(f"Generated {len(all_events)} events across all decades")
    
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

# ---- Add missing columns with default values ----
def ensure_all_columns_exist(df):
    """Makes sure all required columns exist in the dataframe, adding empty ones if needed"""
    required_columns = [
        "Event", "Year(s)", "Region", "Affected Segment(s)", "Impact Summary", 
        "Event Category", "Impact Level", "Affected Commodity", "Response Mechanism",
        "Response Type", "Start Month", "End Month", "Estimated Duration", 
        "Risk Type", "Insurance Impact", "Trade Routes Affected", 
        "Positive Outcome / Opportunity", "Source / Reference Link",
        "AI Event Summary", "RAG_Context", "Chokepoint (if any)", "Cluster",
        "Duration_Months", "Cluster Label", "Start Year", "Decade", "Season",
        "Trend Flag", "Risk Score", "lat", "lon", "Anomaly_ZScore",
        "Anomaly_IsoForest", "Anomaly_Detected", "Score_Risk", "Score_Chokepoint",
        "Score_Season", "Score_RegionAnomaly", "Score_ClusterAnomaly",
        "Warning Score", "Warning Level", "Trigger Reason"
    ]
    
    # Add any missing columns with empty values
    for col in required_columns:
        if col not in df.columns:
            # Choose appropriate default value based on column name
            if col in ["Score_Risk", "Score_Chokepoint", "Score_Season", 
                      "Score_RegionAnomaly", "Score_ClusterAnomaly", "Warning Score",
                      "lat", "lon", "Anomaly_ZScore", "Risk Score", "Duration_Months"]:
                df[col] = 0
            elif col in ["Anomaly_IsoForest"]:
                df[col] = 0.0
            elif col in ["Anomaly_Detected", "Trend Flag"]:
                df[col] = "No"
            elif col in ["Warning Level"]:
                df[col] = "Low"
            else:
                df[col] = ""
    
    return df

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
                "consistency_seed": consistency_seed if use_seed else "Not used",
                "max_api_calls": max_api_calls,
                "parallel_workers": parallel_workers,
                "batch_size": batch_size,
                "model": model_name
            }
        })

if generate_button:
    try:
        # Generate events with decade focus
        st.subheader("Step 1: Generating Events By Decade")
        
        # Calculate how many events we aim to get per decade
        num_decades = 7  # 1960s through 2020s
        events_per_decade = max(5, dataset_size // num_decades)
        total_target = events_per_decade * num_decades
        
        st.info(f"🔄 Targeting approximately {events_per_decade} events per decade across 7 decades")
        
        events = generate_events_with_decades(
            total_size=dataset_size,
            workers=parallel_workers,
            batch_size=batch_size,
            consistency_seed=consistency_seed if use_seed else None,
            max_api_calls=max_api_calls
        )
        
        if not events:
            st.error("Failed to generate any events. Please try again or adjust settings.")
        else:
            # Convert to DataFrame
            df = pd.DataFrame(events)
            
            # Ensure all requested columns exist
            df = ensure_all_columns_exist(df)
            
            st.success(f"✅ Generated {len(df)} events successfully!")
            
            # Process in chunks
            st.subheader("Step 2: Calculating Risk Scores")
            df = process_dataframe_in_chunks(df, weights, scoring_config, chunk_size)
            
            # Analyze decade distribution
            st.subheader("Decade Distribution")
            if 'Decade' in df.columns:
                decade_counts = df['Decade'].value_counts().sort_index()
                st.bar_chart(decade_counts)
                
                # Show decade summary
                decade_col1, decade_col2 = st.columns([1, 2])
                with decade_col1:
                    st.write("Events by Decade:")
                    decade_summary = pd.DataFrame({
                        'Decade': decade_counts.index,
                        'Events': decade_counts.values,
                        'Percentage': (decade_counts.values / len(df) * 100).round(1)
                    })
                    st.dataframe(decade_summary)
                
                with decade_col2:
                    # Calculate coverage
                    missing_decades = []
                    all_decades = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
                    for decade in all_decades:
                        if decade not in decade_counts.index:
                            missing_decades.append(decade)
                    
                    if missing_decades:
                        st.warning(f"⚠️ Missing data for {len(missing_decades)} decades: {', '.join(missing_decades)}")
                        st.info("Try increasing 'Maximum API Calls' to improve decade coverage.")
                    else:
                        st.success("✅ Data includes events from all decades (1960s-2020s)")
            
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
                # Excel download
                excel_io = io.BytesIO()
                with pd.ExcelWriter(excel_io, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name="Maritime Events")
                excel_io.seek(0)
                
                st.download_button(
                    "📥 Download Excel",
                    data=excel_io,
                    file_name="maritime_events.xlsx",
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

# Add basic visualization if data exists
if 'maritime_events_df' in st.session_state:
    st.subheader("Data Visualization")
    viz_df = st.session_state['maritime_events_df']
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Warning Levels", "Impact Analysis", "Regional Distribution", "Decade Analysis"])
    
    with viz_tab1:
        # Warning Level Distribution
        if 'Warning Level' in viz_df.columns:
            st.write("Warning Level Distribution:")
            warning_counts = viz_df['Warning Level'].value_counts()
            st.bar_chart(warning_counts)
    
    with viz_tab2:
        # Impact Level Distribution
        if 'Impact Level' in viz_df.columns:
            st.write("Impact Level Distribution:")
            impact_counts = viz_df['Impact Level'].value_counts()
            st.bar_chart(impact_counts)
            
        # Response Mechanism Distribution
        if 'Response Mechanism' in viz_df.columns:
            st.write("Response Mechanism Distribution:")
            response_counts = viz_df['Response Mechanism'].value_counts()
            st.bar_chart(response_counts)
    
    with viz_tab3:
        # Regional Distribution
        if 'Region' in viz_df.columns:
            st.write("Regional Distribution:")
            region_counts = viz_df['Region'].value_counts().head(10)  # Top 10 regions
            st.bar_chart(region_counts)
            
        # Chokepoint Distribution
        if 'Chokepoint (if any)' in viz_df.columns:
            st.write("Chokepoint Distribution:")
            # Filter out empty values
            chokepoint_data = viz_df[viz_df['Chokepoint (if any)'].astype(str).str.strip() != '']
            if not chokepoint_data.empty:
                chokepoint_counts = chokepoint_data['Chokepoint (if any)'].value_counts().head(10)
                st.bar_chart(chokepoint_counts)
            else:
                st.info("No chokepoint data available.")
                
    with viz_tab4:
        # Decade Analysis
        if 'Decade' in viz_df.columns:
            st.write("Events by Decade:")
            decade_counts = viz_df['Decade'].value_counts().sort_index()
            st.bar_chart(decade_counts)
            
            # Show decade breakdown with warning levels
            st.write("Decade Breakdown by Warning Level:")
            decade_warning = pd.crosstab(viz_df['Decade'], viz_df['Warning Level']).sort_index()
            st.dataframe(decade_warning)
            
            # Decade timeline of events
            if 'Start Year' in viz_df.columns:
                st.write("Timeline of Events:")
                # Convert to numeric if not already
                try:
                    viz_df['Start Year'] = pd.to_numeric(viz_df['Start Year'], errors='coerce')
                    year_counts = viz_df['Start Year'].value_counts().sort_index()
                    st.line_chart(year_counts)
                except Exception as e:
                    st.warning(f"Could not generate timeline: {str(e)}")
                
    # Add a download button for full column set
    st.subheader("Column Selection for Download")
    
    # Group columns into categories for better organization
    column_categories = {
        "Basic Information": ["Event", "Year(s)", "Region", "Start Year", "Decade", "Season"],
        "Impact Details": ["Impact Level", "Impact Summary", "Affected Segment(s)", "Affected Commodity", 
                          "Trade Routes Affected", "Chokepoint (if any)"],
        "Response Information": ["Response Mechanism", "Response Type", "Positive Outcome / Opportunity"],
        "Timing and Duration": ["Start Month", "End Month", "Estimated Duration", "Duration_Months"],
        "Risk and Warning": ["Risk Score", "Risk Type", "Warning Score", "Warning Level", "Trigger Reason"],
        "Classification": ["Event Category", "Cluster", "Cluster Label"],
        "Anomaly Detection": ["Anomaly_ZScore", "Anomaly_IsoForest", "Anomaly_Detected", "Trend Flag"],
        "Score Components": ["Score_Risk", "Score_Chokepoint", "Score_Season", "Score_RegionAnomaly", "Score_ClusterAnomaly"],
        "Other": ["lat", "lon", "Insurance Impact", "AI Event Summary", "RAG_Context", "Source / Reference Link"]
    }
    
    # Create expandable sections for each category
    selected_columns = []
    for category, cols in column_categories.items():
        with st.expander(category, expanded=(category == "Basic Information")):
            # Filter to only include columns that exist in the dataframe
            available_cols = [col for col in cols if col in viz_df.columns]
            if available_cols:
                selected = st.multiselect(
                    f"Select {category} columns",
                    options=available_cols,
                    default=available_cols
                )
                selected_columns.extend(selected)
    
    # Custom download with selected columns
    if selected_columns:
        selected_df = viz_df[selected_columns]
        
        st.subheader("Download Custom Dataset")
        custom_cols = st.columns(2)
        
        with custom_cols[0]:
            # Excel download
            excel_io = io.BytesIO()
            with pd.ExcelWriter(excel_io, engine='openpyxl') as writer:
                selected_df.to_excel(writer, index=False, sheet_name="Maritime Events")
            excel_io.seek(0)
            
            st.download_button(
                "📥 Download Custom Excel",
                data=excel_io,
                file_name="custom_maritime_events.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with custom_cols[1]:
            # CSV download
            csv_io = io.BytesIO()
            selected_df.to_csv(csv_io, index=False)
            csv_io.seek(0)
            
            st.download_button(
                "📥 Download Custom CSV",
                data=csv_io,
                file_name="custom_maritime_events.csv",
                mime="text/csv"
            )