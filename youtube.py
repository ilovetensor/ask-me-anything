from youtube_transcript_api import YouTubeTranscriptApi


def extract_id(link):
   return link[-11:]

def raw_text_from_link(link):
   video_id = extract_id(link)
   transcript = YouTubeTranscriptApi.get_transcript(video_id) 
   raw_text=""
   for i in transcript:
      raw_text += i['text'] + " "
   return raw_text

link="https://youtu.be/ouYqJ_Y0mBw"
raw_text = raw_text_from_link(link)
print(raw_text)