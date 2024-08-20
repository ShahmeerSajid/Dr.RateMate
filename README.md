## Dr.Rate Mate (What does it do?)

Dr. RateMate is your clever sidekick in the often confusing world of course and professor selection. Built on a robust machine learning model, it’s trained on over 850 authentic reviews from students, Rate My Prof, Reddit, and even a few random comments we picked up along the way. This AI-driven tool dives deep into the chaos to deliver ratings that you can actually rely on—no more guesswork, just straightforward insights.

Here’s how it works: You feed Dr. RateMate a review, and it gets to work, analyzing every detail to give you both course and professor ratings on a scale of -1 to 5. But there’s a catch—if you forget to mention the course or the professor, you’ll be hit with a -1 for whichever detail is missing. It’s like the AI’s way of saying, “Come on, give me something to work with!”

This model isn’t just about crunching numbers—it’s about making your academic life easier. Dr. RateMate turns a jumble of opinions into clear, data-driven ratings, so you can make informed decisions with confidence. Whether you’re hunting for the perfect class or trying to avoid that professor with the brutal exams, Dr. RateMate is your go-to guide. Think of it as your personal course and professor detective—minus the magnifying glass, but with all the smarts and none of the drama. So, dive in, and let Dr. RateMate help you ace the art of course selection, one review at a time!

**--->** For a detailed explanation of Dr.RateMate, please check this out: https://devpost.com/software/dr-ratemate

## How to run the project?
After setting up your Python environment (latest Python version is recommended), follow these steps to get everything up and running smoothly:

- Clone the repository: **git clone https://github.com/ShahmeerSajid/Dr.RateMate.git**
- Navigate to the clond repository: **cd StockShockWaveWizard**
- Install Dependencies:
  On Terminal (for Mac) or Powershell (for windows), run the commannd **pip install -r requirements.txt**
- Create Your Own config.py:
  The project requires certain configuration settings, such as an OpenAI API key, which are not included in this repository for security reasons. You'll need to create a file named config.py in the project directory and include your own OpenAI API key or any other necessary configuration. An example config.py might look like this:
  **My_API_key = "your-openai-api-key-here"**
  
  Then in mainNLPModel.py, replace Shahmeer_OpenAI_API_Key with My_API_key.  --> Important
  
- The project includes pre-trained models stored in the .pkl files (stacked_clf_Course Rating.pkl and stacked_clf_Prof Rating.pkl).
  These files contain the trained algorithms and must be present in the project directory for the code to run successfully. By using these files, you won't need to retrain the models, which saves significant time and computational resources.
- With all dependencies installed, and your 'config.py' in place, (on Terminal --> Mac or Powershell --> Windows), you can execute the project by : **streamlit run mainNLPModel.py**

**This setup guarantees your project runs smoothly without making you retrain those stubborn models—because who has time for that? Plus, your top-secret API keys will stay as secure as a cat guarding its favorite spot. Just double-check you've got all the files in place before you hit "go"—nobody wants a missing piece to the puzzle! 
Then, sit back, relax, and with these steps, you'll have the website up and running in no time, delivering professor and course ratings straight from your reviews like a pro!**

  
