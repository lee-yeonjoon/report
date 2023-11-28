from konlpy.tag import Okt

class KoreanModel:
    def analyze(self, text):
        okt = Okt()
        tokens = okt.pos(text)
        
        return tokens

korean_analyzer = KoreanModel()
korean_text = "만취 상태로 운전을 하다가 자전거를 탄 시민을 치어 숨지게 하고 달아난 20대가 구속 상태로 재판에 넘겨졌습니다."
korean_text = "창밖에 밤비가 속살거려 육첩방은 남의 나라"
korean_text = "곧 우리 아기 태어난날이 다가와서,나한테 원래 둘째 아들이 있었어"
result = korean_analyzer.analyze(korean_text)
print(result)

import jieba

class ChineseModel:
    def analyze(self, text):
        tokens = jieba.lcut(text)
        
        return tokens


chinese_analyzer = ChineseModel()
chinese_text = "科技部副部长吴朝晖、英伟达首席执行官黄仁勋等科技专家预测：人工智能研发的下一个浪潮是“具身智能”。"
chinese_text = "参观刷身份证就可以进"
chinese_text = "陆建明好像听到了天大的笑话一般大笑起来。"
result = chinese_analyzer.analyze(chinese_text)
print(result)

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class EnglishModel:
    def analyze(self, text):
        # 토큰화
        tokens = word_tokenize(text)

        # 불용어 제거
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

        # 단어 빈도 분포
        freq_dist = FreqDist(filtered_tokens)

        # 가장 빈도가 높은 단어 추출
        common_words = freq_dist.most_common(5)

        return common_words


english_analyzer = EnglishModel()
english_text = "Harbin issued a red blizzard alert – the highest in China’s four-tier warning system – on Sunday and Monday."
english_text = "Being holy day, the beggers shop is shut."
english_text = "And we're not done taking action to get those prices down even more."
result = english_analyzer.analyze(english_text)
print(result)



from langid import classify

class MultilingualTextAnalysis:
    def __init__(self):
        self.language_models = {
            'en': EnglishModel(),
            'ko':KoreanModel(),
            'ch': ChineseModel(),
            
        }

    def analyze_text(self, text):
        lang, _ = classify(text)  # 텍스트의 언어 감지
        if lang in self.language_models:
            return self.language_models[lang].analyze(text)
        else:
            return "Language not supported"
        
        
