import re
import emoji

class Preprocessor:

  # Using hugging face emoji module to change the emojies to words
  def emojiToWord(self, tweet):
      return emoji.demojize(tweet)

  # remove punctuation marks in the tweets
  def remove_punctuations(self,tweet):
      tweet = tweet.replace(':', ' ')     # colon mark
      tweet = tweet.replace('_', ' ')     # underscore
      tweet = tweet.replace('...', ' ')   # three fullstops
      return tweet

  # replacing mentioned usernames with @USER
  def replace_usernames(self,tweet):
      # find place where username is mentioned in the tweet
      users=re.findall(r'[@]\S*', tweet)
      # replace the username with @USER using a loop
      for user in users:
        tweet = tweet.replace(user, '@USER')

      # if there are multiple `@USER` tokens in a tweet, replace it with `@USERS`
      # because some tweets contain so many `@USER` which may cause redundant
      if tweet.find('@USER') != tweet.rfind('@USER'):
          tweet = tweet.replace('@USER', '')
          tweet = '@USERS ' + tweet
      return tweet


  def process_tweet(self,tweet):
    #Remove www.* or https?://*
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))\s+','',tweet)
    tweet = re.sub('\s+((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    #Remove emoji to word
    #https://github.com/carpedm20/emoji
    tweet = self.emojiToWord(tweet)
    #Remove the punctuations
    tweet = self.remove_punctuations(tweet)
    #Remove RTs
    tweet = re.sub('^RT @[A-Za-z0-9_]+: ', '', tweet)
    #Incorrect apostraphe
    tweet = re.sub(r"’", "'", tweet)
    #Replacing @USER
    tweet = self.replace_usernames(tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace ampersands
    tweet = re.sub(r' &amp; ', ' and ', tweet)
    tweet = re.sub(r'&amp;', '&', tweet)
    #Remove emojis
    #tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return str(tweet)
