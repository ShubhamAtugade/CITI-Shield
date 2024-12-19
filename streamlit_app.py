import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from folium.plugins import HeatMap

# Load the datasets
@st.cache_data
def load_visualization_data():
    return pd.read_excel("crp.xlsx")

@st.cache_data
def load_prediction_data():
    return pd.read_excel("new_dataset.xlsx")

visualization_data = load_visualization_data()
prediction_data = load_prediction_data()

# Set layout to wide
st.set_page_config(layout="wide")

# Add custom CSS for horizontal navigation
st.markdown("""
    <style>
        .horizontal-radio > label {
            display: inline-block;
            margin-right: 20px;
            cursor: pointer;
        }
        .centered-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
        }
        .stRadio {
        display: flex;
        justify-content: center;
        gap: 60px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Navigation Bar (Horizontal Radio Buttons)
st.image("logo.png", width=325)
page_options = ["Home", "Visualization", "Prediction", "About Us"]
selected_page = st.radio("", page_options, horizontal=True, key="current_page", label_visibility="collapsed")

# Page Functions
def home_page():
    st.header("Crime Rate in India: An Overview")
    st.write("India, as one of the world's largest and most diverse nations, faces a complex spectrum of crimes that range from petty thefts to organized crime and cybercrimes. According to the National Crime Records Bureau (NCRB), the country's crime rate has seen fluctuations over the years, with urban areas generally recording higher crime rates than rural ones. The most common crimes include theft, robbery, assault, fraud, and crimes against women, such as domestic violence and sexual harassment. Rapid urbanization, economic disparities, and population density contribute significantly to the crime rates in cities like Delhi, Mumbai, and Bengaluru.")
    st.write("Another alarming trend is the rise in cybercrimes as internet penetration deepens across the country. Cases involving financial fraud, identity theft, and data breaches have surged, posing a significant challenge to law enforcement. Crimes against vulnerable populations, such as children and senior citizens, remain a critical issue, highlighting the need for better protection mechanisms. At the same time, a substantial portion of crimes remains unreported due to social stigma, lack of trust in the police, or fear of repercussions, which makes addressing the actual crime rate even more challenging.")
    st.write("India, as one of the world's largest and most diverse nations, faces a complex spectrum of crimes, ranging from petty thefts to organized crime and cybercrimes. According to the National Crime Records Bureau (NCRB), the country's crime rate has fluctuated over the years, with urban areas generally recording higher crime rates than rural ones. Common crimes include theft, robbery, assault, fraud, and crimes against women, such as domestic violence and sexual harassment. Factors such as rapid urbanization, economic disparities, and population density significantly contribute to the crime rates in major cities like Delhi, Mumbai, and Bengaluru. Additionally, the rise in cybercrimes, including financial fraud, identity theft, and data breaches, has become a growing challenge for law enforcement as internet usage continues to expand")
def visualization_page():
    st.header("Crime Data Visualization")
    st.write("### Crime Data Table")
    st.dataframe(visualization_data, height=425)

    st.write("### Crime Data Bar Chart and Heatmap")

    # Bar Chart
    crime_types = visualization_data.select_dtypes(include=['number']).columns.tolist()
    if crime_types:
        selected_crime_type = st.selectbox("Select Crime Type for Bar Chart", options=crime_types)
        if selected_crime_type:
            fig, ax = plt.subplots()
            ax.barh(visualization_data['City'], visualization_data[selected_crime_type], color='#FF4B4B', alpha=0.85)
            ax.set_xlabel("Count", color='white', fontsize=12)
            ax.set_ylabel("City", color='white', fontsize=12)
            ax.set_title(f"Horizontal Bar Chart: City vs {selected_crime_type}", color='white', fontsize=14)
            ax.tick_params(colors='white', labelsize=12)
            fig.set_size_inches(8, 7)
            fig.patch.set_alpha(0.0)  # Transparent background
            ax.set_facecolor('none')

    # Heatmap Setup
    data = visualization_data.copy()

    # Add latitude and longitude columns for cities (mock data for demonstration)
    city_coordinates = {
        "Ahmedabad": [23.0225, 72.5714],
        "Bengaluru": [12.9716, 77.5946],
        "Chennai": [13.0827, 80.2707],
        "Coimbatore": [11.0168, 76.9558],
        "Delhi": [28.7041, 77.1025],
        "Ghaziabad": [28.6692, 77.4538],
        "Hyderabad": [17.3850, 78.4867],
        "Indore": [22.7196, 75.8577],
        "Jaipur": [26.9124, 75.7873],
        "Kanpur": [26.4499, 80.3319],
        "Kochi": [9.9312, 76.2673],
        "Kolkata": [22.5726, 88.3639],
        "Kozhikode": [11.2588, 75.7804],
        "Lucknow": [26.8467, 80.9462],
        "Mumbai": [19.0760, 72.8777],
        "Nagpur": [21.1458, 79.0882],
        "Patna": [25.5941, 85.1376],
        "Pune": [18.5204, 73.8567],
        "Surat": [21.1702, 72.8311],
    }

    data["Latitude"] = data["City"].map(lambda x: city_coordinates[x][0] if x in city_coordinates else None)
    data["Longitude"] = data["City"].map(lambda x: city_coordinates[x][1] if x in city_coordinates else None)

    # Filter out rows without coordinates
    data = data.dropna(subset=["Latitude", "Longitude"])

    # Reshape the data for visualization
    data_long = data.melt(
        id_vars=["Year", "City", "Latitude", "Longitude"], 
        value_vars=[
            "Murder", "Kidnapping", "Crime against women", 
            "Crime against children", "Crime Committed by Juveniles", 
            "Crime against Senior Citizen", "Crime against SC", 
            "Crime against ST", "Economic Offences", "Cyber Crimes"
        ],
        var_name="Crime Type",
        value_name="Count"
    )

    # Filter controls
    years = st.multiselect("Select Year(s)", options=sorted(data_long["Year"].unique()), default=sorted(data_long["Year"].unique()))
    crime_types = st.multiselect("Select Crime Type(s)", options=data_long["Crime Type"].unique(), default=data_long["Crime Type"].unique())

    # Filter data based on selections
    filtered_data = data_long[(data_long["Year"].isin(years)) & (data_long["Crime Type"].isin(crime_types))]

    if not filtered_data.empty:
        map_center = [filtered_data["Latitude"].mean(), filtered_data["Longitude"].mean()]
        m = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB dark_matter")

        # Prepare data for HeatMap
        heat_data = filtered_data.loc[:, ["Latitude", "Longitude", "Count"]].values.tolist()
        HeatMap(heat_data).add_to(m)

        # Display map
        container = st.container()
        with container:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig)
            with col2:
                folium_static(m)
    else:
        st.write("No data available to display on the map.")

def train_model():
    data = load_prediction_data()
    le_city = LabelEncoder()
    le_type = LabelEncoder()
    data['city_encoded'] = le_city.fit_transform(data['City'])
    data['type_encoded'] = le_type.fit_transform(data['Type'])
    X = data[['city_encoded', 'type_encoded']]
    y = data['Crime Rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    return model, le_city, le_type, mae, r2

model, le_city, le_type, mae, r2 = train_model()


def prediction_page():
    st.header("Crime Rate Prediction")
    st.write("### Predict Future Crime Trends")
    cities = list(le_city.classes_)
    crimes = list(le_type.classes_)
    selected_city = st.selectbox("Select City", cities)
    selected_crime = st.selectbox("Select Crime Type", crimes)
    if st.button("Submit"):
        city_encoded = le_city.transform([selected_city])[0]
        type_encoded = le_type.transform([selected_crime])[0]
        input_data = pd.DataFrame([[city_encoded, type_encoded]], columns=['city_encoded', 'type_encoded'])
        predicted_rate = model.predict(input_data)[0]
        trend = "increasing" if predicted_rate > 0 else "decreasing"
        likelihood = abs(predicted_rate) * 100
        st.write(f"### Results")
        st.write(f"**Selected City:** {selected_city}")
        st.write(f"**Selected Crime:** {selected_crime}")
        st.write(f"**Prediction:** Crime rate is likely to be {trend} with a likelihood of {likelihood:.2f}%.")
    st.write("### Model Evaluation")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"R² Score: {r2:.2f}")

def about_page():
    st.header("About Citi-Shield")
    st.write("Citi-Shield is designed to provide insights into crime data for better urban safety and planning.")
    st.write("The CITI-Shield Crime Rate Analysis and Prediction Model is an application that forecasts crime rates in 19 Indian metropolitan areas using machine learning techniques. In order to lower crime rates and enhance public safety, this initiative aims to help researchers, law students, and the defense industry comprehend crime trends and allocate resources efficiently.")
    st.write("The CITI-Shield crime rate analysis and prediction model represents an invaluable asset for researchers, law enforcement agencies, and defense sectors. Its primary objective is to identify crime patterns and forecast potential future criminal activities. By predicting areas where crime may escalate, these entities can strategically deploy resources to locations anticipated to experience increases, thereby promoting a proactive approach that contributes to the reduction of overall crime rates and the enhancement of public safety.")
    st.write("The foundational dataset for this analysis has been rigorously compiled from publications accessible on the Indian National Crime Records Bureau (NCRB) official website. It encompasses crime statistics from 19 metropolitan cities over the years 2014 to 2021. This application enables predictions of crime rates across ten distinct categories, which include murder, kidnapping, crimes against women, crimes against children, juvenile offenses, crimes against senior citizens, crimes targeting Scheduled Castes (SC), crimes against Scheduled Tribes (ST), economic offenses, and cybercrimes.")
    st.write("To generate its predictions, the system employs the Random Forest Regression model from the scikit-learn library. This ensemble learning technique is proficient in predicting continuous values based on provided inputs, including the year, city name, and type of crime. The model functions by constructing multiple decision trees, each contributing to the prediction for the target variable. The final output is computed by averaging these individual predictions, resulting in enhanced accuracy compared to a single decision tree methodology. Notably, the model has attained an accuracy rate of 93.20% on the testing dataset when forecasting crime rates.")
    
    st.write("This model is build by Shubham Atugade which includes data cleaning, data analysis, data formating and machine learning model to analysis and prediction of crimes against every crime type happening in indian cities.")
# Render the current page
if selected_page == "Home":
    home_page()
elif selected_page == "Visualization":
    visualization_page()
elif selected_page == "Prediction":
    prediction_page()
elif selected_page == "About Us":
    about_page()
