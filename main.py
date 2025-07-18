# main.py
import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from travel_tools import get_flights, suggest_hotels

# Load environment variables
load_dotenv()
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
config = RunConfig(model=model, tracing_disabled=True)

destination_agent = Agent(
    name="DestinationAgent",
    instructions="You recommend travel destinations based on user's preferences.",
    model=model,
)

booking_agent = Agent(
    name="BookingAgent",
    instructions="You handle travel bookings including flights and hotels.",
    model=model,
    tools=[get_flights, suggest_hotels],
)

explore_agent = Agent(
    name="ExploreAgent",
    instructions="You suggest food and places to explore in the destination.",
    model=model,
)  

def main():
    print("\Welcome to the AI Travel Designer\n")
    mood = input("What is your mood for travel? (e.g., adventure, relaxation, cultural): ")

    result1 = Runner.run_sync(destination_agent, mood, run_config=config)
    destination = result1.final_output.strip()
    print(f"destination suggested:", destination)

    result2 = Runner.run_sync(booking_agent, destination, run_config=config)
    print(f"Booking details:", result2.final_output.strip)

    result3 = Runner.run_sync(explore_agent, destination, run_config=config)
    print(f"Explore suggestions:", result3.final_output)\
    
    if __name__ == "__main__":
        main()
        print("Thank you for using the AI Travel Designer!")
        print("Safe travels!")
        print("For more information, visit our website or contact support.")
        print("Follow us on social media for travel tips and updates.")
        print("We hope you have a wonderful travel experience!")
        print("Goodbye!")
        print("Remember to share your travel stories with us!")
        print("Your feedback is valuable to us!")
        print("Stay tuned for more features and updates in the future!")
        print("Thank you for choosing our service!")
        print("We appreciate your business and look forward to serving you again!")
        print("Have a great day!")