# **Product Analyzer**

The **Product Analyzer** is a comprehensive pipeline that extracts tweets related to a specific product or keyword, processes the collected data, analyzes the sentiments of the tweets, and visualizes the results. The primary goal of this project is to identify public opinions and trends surrounding a given topic, which is achieved through the integration of **web scraping**, **text processing**, **sentiment analysis**, and **data visualization**. This project demonstrates a real-world application of **Natural Language Processing (NLP)**, data analysis, and software engineering principles. It provides valuable insights into how users on Twitter feel about specific products, events, or general topics.

The project is broken down into three key components, each implemented within its own class: `Scrape`, `DataCleaning`, and `SentimentAnalysis`. 

### **1. Scrape Class**
The `Scrape` class handles the connection to the **Tweepy API** for fetching tweets. By searching for a specified keyword or phrase, the class extracts relevant information, such as:
- User’s name
- Date of the tweet
- Number of likes
- Source of the tweet
- Full text of the tweet

All of this information is stored in a **Pandas DataFrame** for further processing. This modular approach ensures that the scraping functionality remains flexible and reusable for any keyword or dataset.

### **2. DataCleaning Class**
Once the tweets have been collected, the `DataCleaning` class processes the raw text data to ensure accuracy and consistency. This involves:
- Removing URLs
- Eliminating stop words, punctuation, and special characters
- Stripping unnecessary white spaces

The cleaned tweets are added as a new column in the original DataFrame. The cleaning process is crucial because text data from social media often includes noise, such as hyperlinks or filler words, that can skew sentiment analysis results. The class design emphasizes simplicity while ensuring that the output data is reliable and ready for analysis.

### **3. SentimentAnalysis Class**
The cleaned dataset is then passed to the `SentimentAnalysis` class, which performs sentiment analysis using the **TextBlob** library. TextBlob was selected for its ease of integration and simplicity in determining sentiment polarity. The sentiment scores range from -1 to 1:
- **Negative values** represent negative sentiments
- **Positive values** indicate positive sentiments
- A score of **0** signifies a neutral sentiment

Additionally, the class categorizes tweets into three sentiment categories: Positive, Negative, and Neutral, which are added as new columns in the DataFrame. To enhance the interpretability of the results, the `SentimentAnalysis` class generates two visualizations:
- A **histogram** to show the distribution of sentiment scores
- A **bar chart** summarizing the sentiment categories

These visualizations are saved as PNG files and offer a clear and intuitive understanding of the analysis.

### **Testing and Validation**
The project includes a robust test suite implemented in `test_project.py` to validate the functionality of all major components. The tests are written using the **pytest** framework and include unit tests for data cleaning, sentiment analysis, and tweet scraping. Mock-based testing is used extensively to simulate external dependencies like the **Tweepy API** and **Matplotlib’s** plotting functions. For example:
- The **Twitter API** is mocked to avoid live data dependency and rate-limiting issues.
- **Matplotlib's** `savefig()` is mocked to validate that visualizations are generated without requiring file creation.

The tests ensure that:
- Sentiment analysis correctly categorizes text
- Cleaned tweets are accurate
- Visualizations are generated and saved as expected

### **Design Choices**
Several design choices were carefully considered during development:
- **TextBlob** was chosen for sentiment analysis due to its simplicity, though alternatives like **VADER** or **spaCy** were considered for more advanced use cases.
- A **class-based design** was adopted to ensure modularity, with each class handling a single responsibility. This enhances code readability, maintainability, and reusability.
- **Mocking external dependencies** for testing ensures reliability and repeatability without requiring live API calls or file generation.

These choices emphasize flexibility, scalability, and robustness, making the project suitable for further extension or adaptation.

### **Setup and Execution**
To run this project, users must first set up the required dependencies. Install the necessary libraries using the following command:

```bash
pip install pandas tweepy textblob matplotlib pytest
```

### **API Configuration**
Users must configure their Twitter API credentials to enable data scraping. The `Scrape` class will use these credentials to authenticate and fetch tweets.

### **Execution**
Once the environment is set up, running the project will:
1. Fetch tweets related to a specified keyword.
2. Clean the raw tweet data.
3. Analyze the sentiment of each tweet and categorize it.
4. Generate and save visualizations summarizing the sentiment analysis.

The output will include a cleaned dataset, sentiment analysis results, and visualizations saved as PNG files.

### **Conclusion**
This project is a valuable tool for analyzing public opinion trends on Twitter. Whether for market research, social studies, or product reviews, the sentiment analysis pipeline provides actionable insights backed by reliable data and clear visualizations. By combining web scraping, text processing, and sentiment analysis, this project highlights the power of data-driven analysis in understanding how people express their opinions online.

With a modular design, extensive testing, and thoughtful design choices, this project serves as an excellent demonstration of applied data analysis and NLP techniques.
