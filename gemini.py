import pandas as pd
import google.generativeai as genai
import json
import re
import time

# Define the path to the saved DataFrame in Google Drive
input_path = './processed_reddit_stories.csv'

# Load the DataFrame
df = pd.read_csv(input_path)

# Display the first few rows to confirm it's loaded
print(df.head())

# Filter stories with similarity score above the threshold (e.g., 0.5)
relevant_stories = df[df['similarity'] >= 0.5]

# Assuming 'relevant_stories' DataFrame is already created from the previous step

question_context = """

I want you to analyse the reddit story I just sent you, the goal is to give me back with some informations:
- Firstly I want to get the profile of the user : age of the couple, supposed income range, living with nuclear or extented family
- Secondly I want to categorize the story into multiple issue.
Please provide your answer as a json filing the following template :
{
  "age_group":"supposed age here",
  "income_group":"low income|medium income|high income",
  "family_type":"nuclear|extented",
  "number_child":"number of child",
  "categorisation":{
    "communication_issue":{
      "attention":"score from one to ten about lack of giving attention between member",
      "miscommunication":"score from one to ten about lack of communication"
    },
    "society":{
      "mass_media":"score from one to ten about the pressure experienced from the social network, perfect instagram life...",
      "personnal_circle":"score from one to ten about the pressure experienced from the distance with friend family...",
      "career":"score from one to ten about the pressure experience by income, career change..."
    },
    "private_life":{
      "natural_gender_gap":"score from one to ten about the physiological difference experienced by the couple",
      "emotional_inteligence":"score from one to ten about the miscomprehension of the other one gender"
    }
  }
}

YOU HAVE ONLY RIGHT TO ANSWER A JSON NO HEADERS, NO EXPLANATION, NO TEXT, JUST JSON.
"""

# Import the necessary libraries for Gemini API


# Configure the Gemini API (assuming your API key is stored in Colab secrets as 'GOOGLE_API_KEY')
GOOGLE_API_KEY="AIzaSyDFjx8PtbtMY4wPVeR5mod4oC3_TCNbMpQ"
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Generative Model (replace 'gemini-pro' with the desired model)
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')

result = []

# Iterate through the top 250 stories
for index, story in relevant_stories.iterrows():

    if index < 0 :
        continue

    story_title = story['title']
    story_text = story['selftext']

    prompt = "START_OF_REDDIT_STORY\n\n" + str(story_text) + "\n\n END_OF_REDDIT_STORY" + question_context

    # TODO: Add your Gemini API call here
    # Example:
    # prompt = f"Analyze the following story and provide insights on [your question]:\nTitle: {story_title}\nStory: {story_text}"
    # response = gemini_model.generate_content(prompt)
    # print(f"Insights for story {index}:\n{response.text}\n")

    # You can access other columns of the story DataFrame row as needed
    # story_similarity = story['similarity']
    # story_cleaned_text = story['cleaned_text']

    #debug
    while True :
      try:
        response= gemini_model.generate_content(prompt)
        break
      except Exception as e:
        print(f"Error generating content for story {index}: {e}")
        time.sleep(5)


    # Process the Gemini response to extract and parse JSON
    try:
        # Remove markdown code block if present
        json_string = re.search(r'```json\n(.*?)```', response.text, re.DOTALL)
        if json_string:
            json_data = json.loads(json_string.group(1))
        else:
            # Assume the response is directly JSON
            json_data = json.loads(response.text)

        relevant_stories.at[index,"gemini_result"] = json_data

        # Now you can work with the extracted json_data dictionary
        print(f"Parsed JSON data for story {index}: {json_data}\n")



    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for story {index}: {e}\nResponse text: {response.text}\n")
    except Exception as e:
        print(f"An unexpected error occurred for story {index}: {e}\nResponse text: {response.text}\n")

    time.sleep(60/30)

df_with_gemini = relevant_stories[relevant_stories['gemini_result'].notna()]

output_path = './processed_gemini_reddit_stories.csv'
df_with_gemini.to_csv(output_path, index=False)