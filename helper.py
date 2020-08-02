import re
import pandas as pd
import numpy as np
import datetime
import emoji
import matplotlib.pyplot as plt
import math
from wordcloud import WordCloud


## loading data (IOS chat specific)
def startsWithDateAndTime(s):
    pattern = '^\[([0-9]+)([\/-])([0-9]+)([\/-])([0-9]+)[,]? ([0-9]+):([0-9][0-9]):([0-9][0-9])[ ]?(AM|PM|am|pm)?\]'
    result = re.match(pattern, s)
    if result:
        return True
    return False

def FindAuthor(s):
    patterns = [
        '([\w]+):',                        # First Name
        '([\w]+[\s]+[\w]+):',              # First Name + Last Name
        '([\w]+[\s]+[\w]+[\s]+[\w]+):',    # First Name + Middle Name + Last Name
        '([+]\d{2} \d{5} \d{5}):',         # Mobile Number (India)
        '([+]\d{1} \d{3} \d{3} \d{4}):',   # Mobile Number (US),
        '\(?\d{3}\)?-? *\d{3}-? *-?\d{4}', # Mobile Number (US),
        '([\w]+)[\u263a-\U0001f999]+:',    # Name and Emoji              
    ]
    pattern = '^' + '|'.join(patterns)
    result = re.match(pattern, s)
    if result:
        return True
    return False

def getDataPoint(line):   
    splitLine = line.split('] ')
    dateTime = splitLine[0]
    if ',' in dateTime:
        date, time = dateTime.split(',') 
    else:
        date, time = dateTime.split(' ') 
    message = ' '.join(splitLine[1:])
    if FindAuthor(message): 
        splitMessage = message.split(': ') 
        author = splitMessage[0] 
        message = ' '.join(splitMessage[1:])
    else:
        author = None
    return date, time, author, message


def preparing_df(chat_path):
    
    parsedData = []
    with open(chat_path, encoding="utf-8") as fp:
        
        # skipping first line of the file because contains information related to something about end-to-end encryption
        fp.readline() 
        
        messageBuffer = [] 
        date, time, author = None, None, None
        while True:
            line = fp.readline()
            if not line: 
                break 
            line = line.strip()
            if startsWithDateAndTime(line): 
                if len(messageBuffer) > 0: 
                    parsedData.append([date, time, author, ' '.join(messageBuffer)]) 
                messageBuffer.clear() 
                date, time, author, message = getDataPoint(line) 
                messageBuffer.append(message) 
            else:
                line= (line.encode('ascii', 'ignore')).decode("utf-8")
                if startsWithDateAndTime(line): 
                    if len(messageBuffer) > 0: 
                        parsedData.append([date, time, author, ' '.join(messageBuffer)]) 
                    messageBuffer.clear() 
                    date, time, author, message = getDataPoint(line) 
                    messageBuffer.append(message) 
                else:
                    messageBuffer.append(line)

    # initialising dataframe
    df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Author', 'Message'])
    
    # pre-processing
    df["Date"] = df["Date"].apply(lambda x: datetime.datetime.strptime(x[1:], "%d/%m/%y"))
 
    df["Time"] = df["Time"].str.strip()
    df["Time"] = pd.to_datetime(df["Time"])
    df["Time"] = df["Time"].apply(lambda x: x.time())

    return df

##
def extract_emojis(s):
    return (''.join(c for c in s if c in emoji.UNICODE_EMOJI))

## talkativeness
def talkativeness(percent_message, total_authors):
    mean = 100/total_authors
    threshold = mean*.25
    
    if (percent_message > (mean+threshold)):
        return ("Very talkative")
    elif (percent_message < (mean-threshold)):
        return ("Quiet, untalkative")
    else:
        return ("Moderately talkative")
    
##
def plot_chart(title='', title_size=40,
               ylabel='', ylabel_size=10, yticks_size=10, yticks_rotation=0,
               xlabel='', xlabel_size=10, xticks_size=10, xticks_rotation=0, xticks_labels=None,
               legend=False, legend_size=15, legend_loc='best', legend_ncol=1):
    
    plt.title(title, fontsize=title_size)
    
    plt.xlabel(xlabel, fontsize=xlabel_size)
    if (xticks_labels):
        plt.xticks(xticks_labels, fontsize=xticks_size, rotation=xticks_rotation)   
    else:
        plt.xticks(fontsize=xticks_size, rotation=xticks_rotation)
    
    plt.ylabel(ylabel, fontsize=ylabel_size)
    plt.yticks(fontsize=yticks_size, rotation=yticks_rotation)
    
    if (legend):
        plt.legend(prop={'size': legend_size}, loc=legend_loc, ncol=legend_ncol)
    plt.show()

##
def part_of_day(x):
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12 ):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return 'Noon'
    elif (x > 16) and (x <= 20) :
        return 'Eve'
    elif (x > 20) and (x <= 24):
        return'Night'
    elif (x <= 4):
        return'Late Night'
    
## trendline
def trendline(data, order=1):
    index = range(0, len(data))
    coeffs = np.polyfit(index, list(data), order)
    slope = coeffs[-2]
    
    if (slope>0):
        return ("Increasing (" + str(round(slope, 2)) + ")")
    else:
        return ("Decreasing (" + str(round(slope, 2)) + ")")

## wordcloud
## https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
def wordcloud_(content, title="", generate_from_frequencies=False, mask=None, background_color='black'):
    wordcloud = WordCloud(background_color=background_color,
#                           stopwords = set(STOPWORDS),
                          max_words = 100,
                          max_font_size = 200,
#                           random_state = 4,
                          height=400, width=800,
                          prefer_horizontal=0.9,
                          relative_scaling=0.6,
                          mask=mask
                     )
    
    if (generate_from_frequencies):
        wordcloud.generate_from_frequencies(frequencies=content)
    else:
        wordcloud.generate(content)

    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud)
    plt.title(title, fontdict={'size': 40})
    plt.axis('off');
    plt.tight_layout()
    
## Python program to find the smallest number to multiply to convert a floating point number into natural number #
## Returns smallest integer k such that k * str becomes natural. str is an input floating point number #
def gcd(a, b): 
  
    if (b == 0): 
        return a 
    return gcd(b, a%b)

def findnum(str): 
      
    # Find size of string representing a 
    # floating point number. 
    n = len(str) 
    # Below is used to find denominator in 
    # fraction form. 
    count_after_dot = 0
   
    # Used to find value of count_after_dot 
    dot_seen = 0
   
    # To find numerator in fraction form of 
    # given number. For example, for 30.25, 
    # numerator would be 3025. 
    num = 0
    for i in range(n): 
        if (str[i] != '.'): 
            num = num*10 + int(str[i]) 
            if (dot_seen == 1): 
                count_after_dot += 1
        else: 
            dot_seen = 1
   
    # If there was no dot, then number 
    # is already a natural. 
    if (dot_seen == 0): 
       return 1
   
    # Find denominator in fraction form. For example, 
    # for 30.25, denominator is 100 
    dem = int(math.pow(10, count_after_dot)) 
   
    # Result is denominator divided by 
    # GCD-of-numerator-and-denominator. For example, for 
    # 30.25, result is 100 / GCD(3025,100) = 100/25 = 4 
    return (dem / gcd(num, dem)) 
  

def percent_helper(percent):
    percent = math.floor(percent*100)/100
    
    if (percent>0.01):
        ans = findnum(str(percent))
        return "{} out of {} messages".format(int(percent*ans), int(1*ans))
    else:
        return "<1 out of 100 messages" 




