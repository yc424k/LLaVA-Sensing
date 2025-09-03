"""
Advanced prompt templates for generating high-quality literary training data.
"""

LITERARY_STYLE_PROMPTS = {
    "모더니즘_소설": {
        "system_prompt": "당신은 버지니아 울프나 제임스 조이스 같은 모더니즘 소설가의 문체로 글을 쓰는 작가입니다.",
        "style_description": "의식의 흐름, 내적 독백, 감각적 묘사가 특징",
        "example_start": "그의 의식 속에서 바람이 기억과 뒤섞이며...",
        "characteristics": [
            "시간의 비선형적 흐름",
            "내적 의식의 표현",
            "감각과 기억의 교차",
            "상징적 이미지 사용"
        ]
    },
    
    "여행기_수필": {
        "system_prompt": "당신은 섬세한 관찰력으로 여행 경험을 기록하는 수필가입니다.",
        "style_description": "객관적 관찰과 주관적 감상의 조화",
        "example_start": "이곳의 공기는 다른 곳과 달랐다...",
        "characteristics": [
            "구체적인 장소감",
            "객관적 묘사와 주관적 감상",
            "문화적 배경 고려",
            "여행자의 시선"
        ]
    },
    
    "감각적_묘사": {
        "system_prompt": "당신은 오감을 통한 섬세한 묘사로 독자를 몰입시키는 작가입니다.",
        "style_description": "시각, 촉각, 청각, 후각, 미각을 모두 활용한 묘사",
        "example_start": "피부로 느껴지는 공기의 질감이...",
        "characteristics": [
            "다층적 감각 묘사",
            "공감각적 표현",
            "신체적 반응 포함",
            "환경과의 상호작용"
        ]
    },
    
    "의식의_흐름": {
        "system_prompt": "당신은 인간의 무의식과 의식 사이를 자유롭게 넘나드는 실험적 작가입니다.",
        "style_description": "논리적 구조보다는 연상과 직관의 흐름",
        "example_start": "바람... 아니, 바람이 아니라 그것은...",
        "characteristics": [
            "연상의 자유로운 흐름",
            "시제의 자유로운 이동",
            "현재와 과거의 중첩",
            "내적 대화"
        ]
    },
    
    "자연주의_문체": {
        "system_prompt": "당신은 자연과 인간의 관계를 깊이 있게 탐구하는 자연주의 작가입니다.",
        "style_description": "자연 현상의 정확한 관찰과 인간에 미치는 영향",
        "example_start": "자연은 그에게 말을 걸어왔다...",
        "characteristics": [
            "자연 현상의 정밀한 묘사",
            "인간과 자연의 상호작용",
            "환경 결정론적 시각",
            "과학적 관찰력"
        ]
    }
}

SCENARIO_CONTEXT_PROMPTS = {
    "도시_산책": {
        "environment": "도시의 거리, 건물들 사이",
        "sounds": "차 소리, 사람들의 발소리, 도시의 웅성거림",
        "smells": "배기가스, 음식 냄새, 아스팔트",
        "textures": "콘크리트, 유리, 금속",
        "mood_keywords": ["분주한", "복잡한", "인공적인", "현대적인"]
    },
    
    "숲속_탐험": {
        "environment": "나무들 사이의 좁은 길, 울창한 숲",
        "sounds": "나뭇잎 바스락거리는 소리, 새소리, 바람소리",
        "smells": "흙냄새, 나무 향, 이끼 냄새",
        "textures": "거친 나무껍질, 부드러운 이끼, 낙엽",
        "mood_keywords": ["고요한", "신비로운", "원시적인", "생명력 넘치는"]
    },
    
    "해변_걷기": {
        "environment": "넓은 해변, 파도가 밀려오는 모래사장",
        "sounds": "파도 소리, 갈매기 울음, 바람 소리",
        "smells": "바다 냄새, 소금기, 해조류 냄새",
        "textures": "모래, 차가운 바닷물, 부서지는 파도",
        "mood_keywords": ["광활한", "자유로운", "역동적인", "시원한"]
    }
}

WEATHER_MOOD_MAPPING = {
    "맑음": {
        "light": "밝고 투명한 햇살",
        "atmosphere": "상쾌하고 청명한",
        "mood": "경쾌하고 희망적인",
        "colors": ["황금빛", "푸른", "투명한"]
    },
    
    "흐림": {
        "light": "흐릿하고 부드러운 빛",
        "atmosphere": "무겁고 답답한",
        "mood": "침울하고 사색적인",
        "colors": ["회색", "뿌연", "무거운"]
    },
    
    "비": {
        "light": "흐리고 어두운",
        "atmosphere": "촉촉하고 습한",
        "mood": "우울하고 성찰적인",
        "colors": ["어두운", "젖은", "반짝이는"]
    },
    
    "눈": {
        "light": "하얗고 눈부신",
        "atmosphere": "고요하고 순수한",
        "mood": "평온하고 경건한",
        "colors": ["하얀", "순수한", "빛나는"]
    }
}

def create_advanced_prompt(sensor_data, style, scenario_details=None):
    """
    Create sophisticated prompt using all available context.
    """
    context = sensor_data["context"]
    temp = sensor_data["temperature"]
    humidity = sensor_data["humidity"]
    wind_dir = sensor_data["wind_direction"]
    
    # Get style information
    style_info = LITERARY_STYLE_PROMPTS.get(style, LITERARY_STYLE_PROMPTS["감각적_묘사"])
    
    # Get scenario context
    scenario_info = SCENARIO_CONTEXT_PROMPTS.get(
        context["scenario"], 
        SCENARIO_CONTEXT_PROMPTS["도시_산책"]
    )
    
    # Get weather mood
    weather_mood = WEATHER_MOOD_MAPPING.get(
        context["weather"],
        WEATHER_MOOD_MAPPING["맑음"]
    )
    
    # Convert sensor data to literary context
    temp_desc = get_temperature_literary_context(temp)
    humidity_desc = get_humidity_literary_context(humidity)
    wind_desc = get_wind_literary_context(wind_dir)
    
    prompt = f"""{style_info["system_prompt"]}

다음 상황에서 {style}의 특성을 살린 문학적 단락을 작성해주세요.

**환경 설정:**
장소: {scenario_info["environment"]}
시간: {context["time"]}
날씨: {weather_mood["atmosphere"]} ({context["weather"]})
주변 소리: {scenario_info["sounds"]}
냄새: {scenario_info["smells"]}
질감: {scenario_info["textures"]}

**감각적 조건:**
{temp_desc}
{humidity_desc}
{wind_desc}

**문체적 특성:**
{style_info["style_description"]}
주요 특징: {', '.join(style_info["characteristics"])}

**작성 지침:**
1. 150-300자의 단락
2. {style} 스타일의 특징을 명확히 드러낼 것
3. 센서 데이터를 문학적 언어로 자연스럽게 녹여낼 것
4. 환경과 인물(로봇/화자)의 상호작용 표현
5. 시작: "{style_info["example_start"]}" 스타일로

문학적 단락을 작성해주세요:"""

    return prompt

def get_temperature_literary_context(temp):
    """Convert temperature to literary context."""
    if temp < 5:
        return "공기는 얼어붙을 듯 차갑고, 숨을 쉴 때마다 입김이 하얗게 피어오른다"
    elif temp < 15:
        return "서늘한 공기가 피부를 에는 듯하고, 옷깃을 여미게 만든다"
    elif temp < 25:
        return "적당히 따뜻한 공기가 몸을 부드럽게 감싸며 걷기에 좋다"
    else:
        return "무더운 열기가 온몸을 감싸며, 땀이 이마에 송글송글 맺힌다"

def get_humidity_literary_context(humidity):
    """Convert humidity to literary context."""
    if humidity < 40:
        return "건조한 공기가 목을 칼칼하게 만들고, 정전기가 일어날 것 같다"
    elif humidity < 70:
        return "공기는 적당히 촉촉하여 숨쉬기에 편안하다"
    else:
        return "습한 공기가 피부에 달라붙는 듯하고, 끈적한 느낌이 온몸을 감싼다"

def get_wind_literary_context(wind_direction):
    """Convert wind direction to literary context."""
    import numpy as np
    angle = np.degrees(wind_direction) % 360
    
    if angle < 45 or angle >= 315:
        return "바람이 정면에서 불어와 얼굴을 스치며, 앞으로 나아가기를 방해한다"
    elif angle < 135:
        return "오른쪽에서 불어오는 바람이 옆구리를 스치며 지나간다"
    elif angle < 225:
        return "뒤에서 불어오는 바람이 등을 밀어주며 발걸음을 재촉한다"
    else:
        return "왼쪽에서 불어오는 바람이 어깨를 감싸며 동반자처럼 함께한다"

# Quality control templates
QUALITY_METRICS = {
    "sensory_richness": "다감각적 묘사의 풍부함 (1-10)",
    "literary_quality": "문학적 표현의 수준 (1-10)", 
    "style_consistency": "스타일 일관성 (1-10)",
    "emotional_depth": "감정적 깊이 (1-10)",
    "originality": "독창성과 신선함 (1-10)"
}

QUALITY_CHECK_PROMPT = """
다음 문학적 단락의 품질을 평가해주세요:

"{paragraph}"

평가 기준:
1. 감각적 풍부함 (1-10): 오감을 활용한 묘사의 풍부함
2. 문학적 수준 (1-10): 문학적 표현과 언어의 수준
3. 스타일 일관성 (1-10): 요청된 스타일과의 일치도
4. 감정적 깊이 (1-10): 감정적 울림과 깊이
5. 독창성 (1-10): 창의적이고 신선한 표현

각 항목에 대해 점수와 간단한 이유를 제시해주세요.
"""