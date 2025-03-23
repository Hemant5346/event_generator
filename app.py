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

# ---- Optimized Scoring Rules with Dynamic Weights ----
def calculate_scores(chunk_df, weights):
    """
    Vectorized calculation of scores for a chunk of the dataframe
    """
    # Initialize score columns
    chunk_df['Score_Risk'] = 0
    chunk_df['Score_Chokepoint'] = 0
    chunk_df['Score_Season'] = 0
    chunk_df['Score_RegionAnomaly'] = 0
    chunk_df['Score_ClusterAnomaly'] = 0
    
    # Convert Risk Score from string to number if needed
    risk_scores = chunk_df.get('Risk Score', pd.Series([0] * len(chunk_df)))
    if risk_scores.dtype == 'object':
        # Handle text values like "High", "Medium", "Low"
        risk_score_map = {"High": 9, "Medium": 5, "Low": 2}
        numeric_risk_scores = risk_scores.map(lambda x: risk_score_map.get(x, 0) if isinstance(x, str) else x)
    else:
        numeric_risk_scores = risk_scores
    
    # Calculate Score_Risk
    chunk_df['Score_Risk'] = np.where(numeric_risk_scores >= 8, 3, 0)
    
    # Calculate Score_Chokepoint
    chokepoints = chunk_df.get('Chokepoint (if any)', pd.Series([''] * len(chunk_df)))
    if 'Chokepoint' in chunk_df.columns and chokepoints.isna().all():
        chokepoints = chunk_df['Chokepoint']
    
    critical_chokepoints = [
        "Strait of Hormuz", "Bab el-Mandeb", "Suez Canal", "Panama Canal", "Taiwan Strait"
    ]
    chunk_df['Score_Chokepoint'] = np.where(
        chokepoints.isin(critical_chokepoints), 2, 0
    )
    
    # Calculate Score_Season
    seasons = chunk_df.get('Season', pd.Series([''] * len(chunk_df)))
    chunk_df['Score_Season'] = np.where(seasons.isin(["Winter", "Summer"]), 1, 0)
    
    # Calculate Score_RegionAnomaly
    regions = chunk_df.get('Region', pd.Series([''] * len(chunk_df)))
    chunk_df['Score_RegionAnomaly'] = np.where(regions.isin(["Middle East", "Adriatic Sea"]), 2, 0)
    
    # Calculate Score_ClusterAnomaly
    clusters = chunk_df.get('Cluster Label', pd.Series([''] * len(chunk_df)))
    critical_clusters = ["Long-Term Geopolitical Conflicts", "Acute Chokepoint or Port Disruptions"]
    chunk_df['Score_ClusterAnomaly'] = np.where(clusters.isin(critical_clusters), 2, 0)
    
    # Handle Impact Level string values
    impact_levels = chunk_df.get('Impact Level', pd.Series([0] * len(chunk_df)))
    if impact_levels.dtype == 'object':
        impact_level_map = {"High": 3, "Medium": 2, "Low": 1}
        numeric_impacts = impact_levels.map(lambda x: impact_level_map.get(x, 0) if isinstance(x, str) else x)
    else:
        numeric_impacts = impact_levels
    
    # Handle Response Mechanism
    response_mechs = chunk_df.get('Response Mechanism', pd.Series([0] * len(chunk_df)))
    if response_mechs.dtype == 'object':
        def map_response(x):
            if not isinstance(x, str):
                return 0
            x_lower = x.lower()
            if "proactive" in x_lower:
                return 1
            elif "reactive" in x_lower:
                return 2
            elif "adaptive" in x_lower or "international" in x_lower:
                return 3
            else:
                return 0
        
        response_values = response_mechs.map(map_response)
    else:
        response_values = response_mechs
    
    # Calculate final scores
    chunk_df['Warning Score'] = (
        weights['impact_level'] * numeric_impacts +
        weights['chokepoint_exposure'] * chunk_df['Score_Chokepoint'] +
        weights['past_incidents'] * (chunk_df['Score_RegionAnomaly'] + chunk_df['Score_ClusterAnomaly']) +
        weights['direct_attack'] * numeric_risk_scores +
        weights['mitigation_difficulty'] * (response_values + chunk_df['Score_Chokepoint'])
    )
    
    # Set Warning Level based on Warning Score
    conditions = [
        chunk_df['Warning Score'] >= 7,
        chunk_df['Warning Score'] >= 4
    ]
    choices = ['High', 'Medium']
    chunk_df['Warning Level'] = np.select(conditions, choices, default='Low')
    
    return chunk_df

# ---- Extract JSON from response with improved error handling ----
def extract_json_from_response(text):
    """Extract JSON array from text that might contain extra content"""
    # Find JSON array pattern
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        potential_json = json_match.group(0)
        try:
            # Try to parse it
            json_data = json.loads(potential_json)
            return json_data
        except json.JSONDecodeError:
            pass
    
    # If that fails, try a more elaborate cleanup
    # Remove markdown code blocks if present
    cleaned_text = re.sub(r'```json\s+|\s+```', '', text)
    cleaned_text = re.sub(r'```\s+|\s+```', '', cleaned_text)
    
    # Try to find JSON array again
    json_match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
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
        except json.JSONDecodeError:
            pass
    
    # If all else fails
    raise ValueError("Could not extract valid JSON from the response")

# ---- Generate batch of events ----
def generate_events_batch(batch_size, base_year, end_year, retries=3):
    """Generate a batch of maritime events using OpenAI API with retry logic"""
    prompt = f"""
    Generate a JSON array of {batch_size} major maritime disruption events from {base_year} to {end_year}.
    Each event should include the following fields in **valid JSON format**:
    Event, Year(s), Region, Affected Segment(s), Impact Summary, Event Category, Impact Level,
    Affected Commodity, Response Mechanism, Response Type, Start Month, End Month,
    Estimated Duration, Risk Type, Insurance Impact, Trade Routes Affected,
    Positive Outcome / Opportunity, Source / Reference Link, AI Event Summary, RAG_Context,
    Chokepoint (if any), Cluster, Duration_Months, Cluster Label, Start Year, Decade,
    Season, Trend Flag, Risk Score, lat, lon, Anomaly_ZScore, Anomaly_IsoForest, Anomaly_Detected
    
    Important: Return ONLY a valid JSON array with no additional text or formatting.
    Do not include markdown code blocks, explanations, or any text outside the JSON array.
    """
    
    for attempt in range(retries):
        try:
            # Try with a system message that strongly enforces JSON-only response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a maritime data generator. Return ONLY valid JSON with no additional text. Do not use markdown code blocks. Do not include explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}  # Force JSON response if API version supports it
            )
            
            # Get the content
            raw_content = response.choices[0].message.content.strip()
            
            try:
                # If response_format worked, this is already a JSON object with a data field
                parsed_data = json.loads(raw_content)
                # Check if it's wrapped in a container object
                if "data" in parsed_data and isinstance(parsed_data["data"], list):
                    return parsed_data["data"]
                elif "maritime_disruption_events" in parsed_data and isinstance(parsed_data["maritime_disruption_events"], list):
                    return parsed_data["maritime_disruption_events"]
                elif isinstance(parsed_data, list):
                    return parsed_data
                else:
                    # Try to extract using regex as fallback
                    return extract_json_from_response(raw_content)
            except json.JSONDecodeError:
                # If direct parsing fails, use our extraction function
                return extract_json_from_response(raw_content)
                
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)  # Wait before retry
                continue
            else:
                raise e
    
    return []

# ---- Parallel event generation ----
def generate_events_parallel(total_size, workers=3, batch_size=10):
    """Generate events in parallel using multiple threads"""
    all_events = []
    batches = [batch_size] * (total_size // batch_size)
    if total_size % batch_size > 0:
        batches.append(total_size % batch_size)
    
    # Create decade ranges for more varied data
    decades = [
        (1960, 1969), (1970, 1979), (1980, 1989), (1990, 1999),
        (2000, 2009), (2010, 2019), (2020, 2024)
    ]
    
    # Distribute batches across decades
    batch_decades = []
    decade_idx = 0
    for _ in batches:
        batch_decades.append(decades[decade_idx])
        decade_idx = (decade_idx + 1) % len(decades)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    completed_batches = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(
                generate_events_batch, 
                batch_size, 
                decade[0], 
                decade[1]
            ): i for i, (batch_size, decade) in enumerate(zip(batches, batch_decades))
        }
        
        # Process results as they complete
        for future in as_completed(future_to_batch):
            try:
                batch_events = future.result()
                all_events.extend(batch_events)
                completed_batches += 1
                progress = completed_batches / len(batches)
                progress_bar.progress(progress)
                status_text.text(f"Generated {len(all_events)} events out of {total_size} ({int(progress*100)}%)")
            except Exception as e:
                st.warning(f"Batch generation error: {str(e)}")
    
    return all_events

# ---- Process data in chunks ----
def process_dataframe_in_chunks(df, weights, chunk_size=1000):
    """Process a large dataframe in chunks to avoid memory issues"""
    result_dfs = []
    
    # Split dataframe into chunks
    chunk_progress = st.progress(0)
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        # Process each chunk
        processed_chunk = calculate_scores(chunk, weights)
        result_dfs.append(processed_chunk)
        
        # Update progress
        chunk_progress.progress((i + 1) / len(chunks))
    
    # Combine results
    return pd.concat(result_dfs, ignore_index=True)

# ---- Streamlit App with Improved UI ----
st.set_page_config(page_title="Maritime Disruption Dataset Generator", layout="wide")
st.title("📊 Maritime Disruption Dataset Generator (1960–Present)")

# --- Weights Input ---
with st.sidebar:
    st.header("Scoring Weights")
    weights = {
        'impact_level': st.slider("Impact Level Weight", 0.0, 5.0, 1.5, 0.1),
        'chokepoint_exposure': st.slider("Chokepoint Exposure Weight", 0.0, 5.0, 1.0, 0.1),
        'past_incidents': st.slider("Past Incidents in Region Weight", 0.0, 5.0, 1.0, 0.1),
        'direct_attack': st.slider("Direct Attack Weight", 0.0, 5.0, 1.0, 0.1),
        'mitigation_difficulty': st.slider("Mitigation Difficulty Weight", 0.0, 5.0, 1.0, 0.1)
    }
    
    st.header("Dataset Size")
    dataset_size = st.slider("Number of events to generate", 10, 1000, 100, 10)
    
    st.header("Performance Options")
    parallel_workers = st.slider("Parallel API calls", 1, 5, 2)
    batch_size = st.slider("Events per API call", 5, 50, 10)
    chunk_size = st.slider("Processing chunk size", 100, 5000, 1000)

# --- Generate Dataset ---
if st.button("🚀 Generate Dataset", type="primary"):
    try:
        # Generate events in parallel with progress tracking
        st.subheader("Step 1: Generating Events")
        events = generate_events_parallel(
            total_size=dataset_size,
            workers=parallel_workers,
            batch_size=batch_size
        )
        
        if not events:
            st.error("Failed to generate events. Please try again.")
        else:
            # Convert to DataFrame
            df = pd.DataFrame(events)
            st.success(f"✅ {len(df)} events generated successfully!")
            
            # Process in chunks with progress tracking
            st.subheader("Step 2: Calculating Risk Scores")
            df = process_dataframe_in_chunks(df, weights, chunk_size)
            
            # Display preview with virtual rendering for large datasets
            st.subheader("Dataset Preview")
            st.dataframe(df, use_container_width=True)
            
            # Provide statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Events", len(df))
            with col2:
                high_risk = len(df[df['Warning Level'] == 'High'])
                st.metric("High Risk Events", high_risk, f"{high_risk/len(df):.1%}")
            with col3:
                regions_count = df['Region'].nunique()
                st.metric("Unique Regions", regions_count)
            
            # Allow downloading in multiple formats
            st.subheader("Download Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # Excel download
                excel_io = io.BytesIO()
                with pd.ExcelWriter(excel_io, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name="Maritime Events")
                excel_io.seek(0)
                
                st.download_button(
                    "📥 Download Excel",
                    data=excel_io,
                    file_name="maritime_events_1960_present.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # CSV download (more efficient for very large datasets)
                csv_io = io.BytesIO()
                df.to_csv(csv_io, index=False)
                csv_io.seek(0)
                
                st.download_button(
                    "📥 Download CSV",
                    data=csv_io,
                    file_name="maritime_events_1960_present.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"Error generating dataset: {str(e)}")
        st.error("Try adjusting the parameters or running again.")