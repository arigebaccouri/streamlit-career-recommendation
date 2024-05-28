import streamlit as st
import joblib
import pandas as pd
from sklearn.exceptions import NotFittedError

# Load the saved models and encoders with correct paths
expanded_rules = joblib.load('/content/expanded_rules.pkl')
xgb_model = joblib.load('/content/xgb_model_default.pkl')
mlb_skills = joblib.load('/content/mlb_skills.pkl')
mlb_primary_intelligence = joblib.load('/content/mlb_primary_intelligence.pkl')
mlb_secondary_intelligence = joblib.load('/content/mlb_secondary_intelligence.pkl')
mlb_bilingual_skills = joblib.load('/content/mlb_bilingual_skills.pkl')
jobs_df = joblib.load('/content/jobs_df.pkl')
feature_names = joblib.load('feature_names.pkl')


# List of available options
skill_options = mlb_skills.classes_
primary_intelligence_options = [
    'social', 'creative', 'linguistic', 'analytical',
    'logical-mathematical', 'practical', 'emotional', 'spatial'
]
secondary_intelligence_options = primary_intelligence_options

# Define the FlexibleCareerRecommendationEngine class
class FlexibleCareerRecommendationEngine:
    def __init__(self, rules):
        self.rules = rules

    def recommend_job_categories(self, skills):
        input_skills = frozenset(skills)
        matching_rules = self.rules[self.rules['antecedents'].apply(lambda x: x.issubset(input_skills))]
        recommended_jobs = set()
        for consequents in matching_rules['consequents']:
            recommended_jobs.update(consequents)
        return recommended_jobs

# Define the CombinedRecommendationEngine class
class CombinedRecommendationEngine:
    def __init__(self, arm_rules, xgb_model):
        self.arm_engine = FlexibleCareerRecommendationEngine(arm_rules)
        self.xgb_model = xgb_model

    def recommend_job_categories(self, skills, primary_intelligence, secondary_intelligence):
        # Get ARM recommendations
        arm_recommendations = self.arm_engine.recommend_job_categories(skills)

        # Prepare the input for the boosting algorithm
        try:
            input_skills = pd.DataFrame([mlb_skills.transform([skills])[0]], columns=mlb_skills.classes_)
            input_primary_intelligence = pd.DataFrame([mlb_primary_intelligence.transform([primary_intelligence])[0]], columns=mlb_primary_intelligence.classes_)
            input_secondary_intelligence = pd.DataFrame([mlb_secondary_intelligence.transform([secondary_intelligence])[0]], columns=mlb_secondary_intelligence.classes_)

            # Combine the input features
            input_features = pd.concat([input_skills, input_primary_intelligence, input_secondary_intelligence], axis=1)
            input_features = input_features.loc[:, ~input_features.columns.duplicated()]  # Remove duplicate columns

            # Ensure the input_features has all the columns expected by the model
            input_features = input_features.reindex(columns=feature_names, fill_value=0)
        except Exception as e:
            st.write(f"Error in preparing input features: {e}")
            return None

        # Get boosting algorithm predictions
        try:
            if not hasattr(self.xgb_model, 'estimators_'):
                raise NotFittedError("The boosting model is not fitted.")
            boosting_predictions = self.xgb_model.predict(input_features)
        except NotFittedError as e:
            st.write(f"Error in boosting model predictions: {e}")
            boosting_predictions = None
        except Exception as e:
            st.write(f"Unexpected error in boosting model predictions: {e}")
            boosting_predictions = None

        # Combine ARM and boosting predictions
        recommended_jobs = set(arm_recommendations)
        if boosting_predictions is not None:
            try:
                for idx, job in enumerate(jobs_df.columns):
                    if boosting_predictions[0, idx] == 1:
                        recommended_jobs.add(job)
            except Exception as e:
                st.write(f"Error in combining recommendations: {e}")

        return recommended_jobs

        # Job categories descriptions
job_descriptions = {
    'UI/UX Designer': 'UI/UX Designers are responsible for creating the visual and interactive aspects of a product. They ensure that the product is user-friendly and engaging.',
    'AI / Machine Learning': 'AI and Machine Learning professionals develop algorithms and models that enable computers to learn and make decisions. This field involves working with large datasets and requires strong programming skills.',
    'Back End Development': 'Back End Developers focus on server-side logic, databases, and application integration. They ensure that the front end of a website or application functions seamlessly.',
    'Data Scientist': 'Data Scientists analyze and interpret complex data to help organizations make informed decisions. They use statistical methods, machine learning, and data visualization techniques.',
    'Project Manager': 'Project Managers plan, execute, and oversee projects to ensure they are completed on time, within scope, and within budget. They coordinate between teams and stakeholders.'
}

# Initialize the combined recommendation engine
combined_engine = CombinedRecommendationEngine(expanded_rules, xgb_model)

# Streamlit app
st.title('Career Recommendation System')

# Display an image at the beginning of the form
st.image("https://images.unsplash.com/photo-1522071820081-009f0129c71c", use_column_width=True)

# Input fields
name = st.text_input('Enter your name:')
skills = st.multiselect('Select your skills:', options=skill_options)

# Option for users to choose if they know their types of intelligence
know_intelligence = st.radio('Do you know your types of intelligence?', ('Yes', 'No'))

if know_intelligence == 'Yes':
    primary_intelligence = st.selectbox('Select your primary intelligence:', options=primary_intelligence_options)
    secondary_intelligence = st.selectbox('Select your secondary intelligence:', options=secondary_intelligence_options)
else:
    st.write("If you don't know your types of intelligence, please follow this link to take a test and find out: [Multiple Intelligences Test](https://www.literacynet.org/mi/assessment/findyourstrengths.html)")
    primary_intelligence = st.selectbox('Select your primary intelligence after taking the test:', options=primary_intelligence_options)
    secondary_intelligence = st.selectbox('Select your secondary intelligence after taking the test:', options=secondary_intelligence_options)

if st.button('Recommend Jobs'):
    recommended_jobs = combined_engine.recommend_job_categories(skills, [primary_intelligence], [secondary_intelligence])

    if recommended_jobs:
        st.success(f'Hi {name}! Your recommended job categories suitable for your profile are:')
        for idx, job in enumerate(recommended_jobs, start=1):
            description = job_descriptions.get(job, 'No description available.')
            st.write(f'{idx}: {job}\n{description}')
          
        # Add a motivational quote and a message in a box
        st.markdown(
            """
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                <p>"The future belongs to those who believe in the beauty of their dreams." - Eleanor Roosevelt</p>
                <p>Remember, finding the right career is a journey. Keep exploring and follow your passion!</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f'Hi {name}, we could not find any job recommendations for your profile.')
