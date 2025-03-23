# Maritime Disruption Dataset Generator

## 📊 Overview
## python3.11
The Maritime Disruption Dataset Generator is a powerful Streamlit application designed to create large, realistic datasets of maritime disruption events from 1960 to the present. It leverages OpenAI's GPT models to generate comprehensive event data with risk scoring and analysis capabilities.

![Maritime Dataset Generator](https://i.imgur.com/placeholder.png)

## 🌟 Features

- **Scalable Data Generation**: Generate from 10 to 1000+ maritime disruption events
- **Parallel Processing**: Utilize multi-threading for faster data generation
- **Memory-Efficient Design**: Process large datasets in chunks to avoid memory issues
- **Comprehensive Event Data**: Each event includes 30+ attributes including regions, impact levels, chokepoints, and more
- **Dynamic Risk Scoring**: Configurable weights for different risk factors
- **Interactive UI**: Real-time progress tracking and statistics
- **Multiple Export Options**: Download as Excel or CSV

## 🚀 Installation

### Prerequisites

- Python 3.7+
- Streamlit
- OpenAI API key

### Dependencies

```bash
pip install streamlit pandas numpy openai openpyxl
```

### Setup

1. Clone this repository or download the script
2. Create a `.streamlit/secrets.toml` file with your OpenAI API key:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## 🔍 Usage

### Configuration Options

The sidebar provides various configuration options:

#### Risk Scoring Weights
- **Impact Level**: Weight for the severity of maritime events
- **Chokepoint Exposure**: Weight for critical maritime chokepoint involvement
- **Past Incidents**: Weight for regions with historical disruptions
- **Direct Attack**: Weight for intentional disruptive actions
- **Mitigation Difficulty**: Weight for response complexity

#### Performance Settings
- **Dataset Size**: Control the number of events to generate (10-1000)
- **Parallel API Calls**: Number of simultaneous API requests (1-5)
- **Events per API Call**: Batch size for each API request (5-50)
- **Processing Chunk Size**: Size of dataframe chunks for efficient processing (100-5000)

### Generating Data

1. Configure your desired settings in the sidebar
2. Click the "🚀 Generate Dataset" button
3. Monitor progress with the real-time progress bars
4. Review the generated data and statistics
5. Download the dataset in your preferred format (Excel or CSV)

## 📋 Data Fields

Each generated maritime disruption event includes the following fields:

- **Event**: Description of the maritime incident
- **Year(s)**: Year(s) when the event occurred
- **Region**: Geographic region where the event took place
- **Affected Segment(s)**: Maritime industry segments impacted
- **Impact Summary**: Brief description of consequences
- **Event Category**: Type of maritime disruption
- **Impact Level**: Severity rating (High/Medium/Low)
- **Affected Commodity**: Goods or resources affected
- **Response Mechanism**: How the event was addressed
- **Response Type**: Category of response
- **Duration**: Temporal details (Start Month, End Month, Estimated Duration)
- **Risk Type**: Category of risk posed
- **Insurance Impact**: Effect on maritime insurance
- **Trade Routes Affected**: Specific shipping lanes impacted
- **Positive Outcome / Opportunity**: Any beneficial results
- **Chokepoint**: Critical maritime passage involved (if any)
- **Geographic Data**: Location information (lat, lon)
- **Anomaly Metrics**: Statistical anomaly indicators
- **Warning Score**: Calculated risk metric
- **Warning Level**: High/Medium/Low classification based on score

## ⚙️ Technical Implementation

### Key Components

1. **Optimized Score Calculation**:
   - Vectorized operations for efficiency
   - Handles string-to-numeric conversions automatically
   - Processes data in chunks to minimize memory usage

2. **Parallel Event Generation**:
   - Uses ThreadPoolExecutor for concurrent API calls
   - Distributes generation across different time periods
   - Includes retry logic for API reliability

3. **Robust JSON Handling**:
   - Multiple strategies for extracting JSON from API responses
   - Fallback mechanisms for handling various response formats
   - Error recovery to maximize successful data generation

4. **Interactive UI Elements**:
   - Real-time progress tracking
   - Informative statistics about generated data
   - Streamlined download options

## 🔧 Advanced Customization

### Modifying Risk Factors

You can customize the risk scoring algorithm by editing the `calculate_scores` function:

- Add new risk factors by creating additional score columns
- Modify the weighting formula in the Warning Score calculation
- Adjust the threshold values for Warning Level classification

### Adding New Data Fields

To incorporate additional data fields:

1. Add the desired fields to the API generation prompt
2. Update the processing logic if needed
3. The application will automatically include new fields in the dataframe and exports

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the application.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Powered by OpenAI's GPT models
- Built with Streamlit
- Developed for maritime risk analysis and simulation

---

*Note: This application generates synthetic data for analysis and educational purposes. It should not be used as the sole source for critical maritime security decisions.*