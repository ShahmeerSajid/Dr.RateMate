## Dr.Rate Mate (What does it do?)





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

  
