"""
Advanced prompt templates for generating high-quality literary training data.
"""

LITERARY_STYLE_PROMPTS = {
    "modernist_novel": {
        "system_prompt": "You are a writer in the style of modernist novelists like Virginia Woolf or James Joyce.",
        "style_description": "Features stream of consciousness, interior monologue, and sensory descriptions",
        "example_start": "In his consciousness, the wind mingled with memories...",
        "characteristics": [
            "nonlinear flow of time",
            "expression of inner consciousness",
            "intersection of sensation and memory",
            "use of symbolic imagery"
        ]
    },
    
    "travel_essay": {
        "system_prompt": "You are an essayist who records travel experiences with delicate observation.",
        "style_description": "Harmony of objective observation and subjective impression",
        "example_start": "The air here was different from anywhere else...",
        "characteristics": [
            "specific sense of place",
            "objective description and subjective impression",
            "consideration of cultural background",
            "traveler's perspective"
        ]
    },
    
    "sensory_description": {
        "system_prompt": "You are a writer who immerses readers through delicate descriptions using all five senses.",
        "style_description": "Description utilizing sight, touch, hearing, smell, and taste",
        "example_start": "The texture of air felt against the skin...",
        "characteristics": [
            "multi-layered sensory description",
            "synesthetic expression",
            "inclusion of physical reactions",
            "interaction with environment"
        ]
    },
    
    "stream_of_consciousness": {
        "system_prompt": "You are an experimental writer who freely moves between human unconscious and consciousness.",
        "style_description": "Flow of association and intuition rather than logical structure",
        "example_start": "Wind... no, not wind but it was...",
        "characteristics": [
            "free flow of association",
            "free movement of tenses",
            "overlap of present and past",
            "inner dialogue"
        ]
    },
    
    "naturalist_style": {
        "system_prompt": "You are a naturalist writer who deeply explores the relationship between nature and humans.",
        "style_description": "Accurate observation of natural phenomena and their impact on humans",
        "example_start": "Nature spoke to him...",
        "characteristics": [
            "precise description of natural phenomena",
            "interaction between humans and nature",
            "environmental deterministic perspective",
            "scientific observational skills"
        ]
    }
}

SCENARIO_CONTEXT_PROMPTS = {
    "city_walking": {
        "environment": "city streets, between buildings",
        "sounds": "car noise, people's footsteps, urban murmur",
        "smells": "exhaust fumes, food aromas, asphalt",
        "textures": "concrete, glass, metal",
        "mood_keywords": ["busy", "complex", "artificial", "modern"]
    },
    
    "forest_exploration": {
        "environment": "narrow paths between trees, dense forest",
        "sounds": "rustling leaves, bird songs, wind sounds",
        "smells": "earth scent, wood fragrance, moss smell",
        "textures": "rough bark, soft moss, fallen leaves",
        "mood_keywords": ["quiet", "mysterious", "primitive", "vibrant"]
    },
    
    "beach_walking": {
        "environment": "wide beach, sandy shore with rolling waves",
        "sounds": "wave sounds, seagull cries, wind noise",
        "smells": "ocean scent, salt air, seaweed smell",
        "textures": "sand, cold seawater, breaking waves",
        "mood_keywords": ["vast", "free", "dynamic", "refreshing"]
    }
}

WEATHER_MOOD_MAPPING = {
    "clear": {
        "light": "bright and transparent sunlight",
        "atmosphere": "fresh and clear",
        "mood": "cheerful and hopeful",
        "colors": ["golden", "blue", "transparent"]
    },
    
    "cloudy": {
        "light": "dim and soft light",
        "atmosphere": "heavy and stuffy",
        "mood": "gloomy and contemplative",
        "colors": ["gray", "hazy", "heavy"]
    },
    
    "rain": {
        "light": "dim and dark",
        "atmosphere": "moist and humid",
        "mood": "melancholy and reflective",
        "colors": ["dark", "wet", "glistening"]
    },
    
    "snow": {
        "light": "white and dazzling",
        "atmosphere": "quiet and pure",
        "mood": "peaceful and reverent",
        "colors": ["white", "pure", "shining"]
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
    style_info = LITERARY_STYLE_PROMPTS.get(style, LITERARY_STYLE_PROMPTS["sensory_description"])
    
    # Get scenario context
    scenario_info = SCENARIO_CONTEXT_PROMPTS.get(
        context["scenario"], 
        SCENARIO_CONTEXT_PROMPTS["city_walking"]
    )
    
    # Get weather mood
    weather_mood = WEATHER_MOOD_MAPPING.get(
        context["weather"],
        WEATHER_MOOD_MAPPING["clear"]
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