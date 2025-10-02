import json
import math
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import glob
import re

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


class SyntheticLiteraryDatasetGenerator:
    """
    Generate synthetic dataset for training sensor-to-literature model.
    Creates pairs of sensor data and corresponding literary paragraphs.
    """
    
    def __init__(
        self,
        api_key: str = None,
        input_dir: str = None,
        use_ollama: bool = True,
        ollama_model: str = "llama3.1:8b",
        use_google_ai: bool = False,
        google_api_key: Optional[str] = None,
        google_model: str = "gemini-2.5-flash-lite"
    ):
        # Model history:
        # - llama3.2:3b (original, Q4_K_M quantization, 2.0GB)
        # - deepseek-r1:8b (4.9GB)
        # - gpt-oss:20b (13GB, too large)
        # - llama3.1:8b (current)
        self.api_key = api_key
        self.input_dir = input_dir
        self.use_ollama = use_ollama
        self.use_google_ai = use_google_ai
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.google_model = google_model
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.google_api_base = "https://generativelanguage.googleapis.com/v1beta"

        if api_key and OPENAI_AVAILABLE and not use_ollama:
            openai.api_key = api_key
            
        # Literary style templates
        self.literary_styles = [
            "modernist_novel",
            "travel_essay", 
            "sensory_description",
            "stream_of_consciousness",
            "naturalist_style"
        ]
        
        # Scenario contexts
        self.scenarios = [
            "city_walking", "forest_exploration", "beach_walking", "mountain_climbing", 
            "park_stroll", "alley_exploration", "riverside_walking", "field_crossing",
            "rooftop_garden", "underground_passage", "bridge_crossing", "plaza_traversal"
        ]
        
        # Time contexts
        self.time_contexts = [
            "dawn", "morning", "forenoon", "noon", "afternoon", "evening", "night", "midnight"
        ]
        
        # Weather contexts
        self.weather_contexts = [
            "clear", "cloudy", "rain", "snow", "fog", "windy", "storm", "shower"
        ]

        self.max_generation_retries = 3

        self.temperature_keywords = {
            "cold": ["cold", "chill", "chilly", "icy", "frost", "cool", "crisp"],
            "mild": ["mild", "temperate", "gentle", "soft", "even"],
            "warm": ["warm", "balmy", "heated", "hot", "sultry", "sweltering", "glowing"]
        }
        self.humidity_keywords = {
            "dry": ["dry", "parched", "arid", "powdery", "brittle"],
            "humid": ["humid", "damp", "moist", "clammy", "sodden", "muggy"]
        }
        self.movement_keywords = {
            "active": ["run", "dash", "sprint", "rush", "charge", "raced"],
            "walking": ["walk", "stroll", "step", "pace", "wander", "footstep", "stride"],
            "still": ["still", "stood", "rest", "quiet", "motionless", "calm", "linger"]
        }

        self.style_guidelines = {
            "modernist_novel": (
                "Embrace stream-of-consciousness narration with intimate emotional detail, "
                "layered imagery, and subtle shifts in perception as the robot reflects on the scene."
            ),
            "travel_novel": (
                "Capture the sense of journey and place through vivid sensory landmarks, "
                "cultural touches, and the narrator's reflective curiosity while moving through the locale."
            ),
        }
    
    def _call_ollama(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Call Ollama API for text generation.

        Args:
            prompt: Text prompt for generation
            temperature: Temperature for generation
            
        Returns:
            str: Generated response
        """
        if not REQUESTS_AVAILABLE:
            raise Exception("requests library not available")
            
        data = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 300
            }
        }
        
        response = requests.post(self.ollama_url, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")

    def _call_google_ai(self, prompt: str, temperature: float = 0.7, max_output_tokens: int = 512) -> str:
        if not REQUESTS_AVAILABLE:
            raise Exception("requests library not available")
        if not self.google_api_key:
            raise Exception("Google API key not configured")

        endpoint = (
            f"{self.google_api_base}/models/{self.google_model}:generateContent"
            f"?key={self.google_api_key}"
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens
            }
        }

        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise Exception("Google AI response missing candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_segments = [part.get("text", "") for part in parts if isinstance(part, dict)]
        result_text = "".join(text_segments).strip()
        if not result_text:
            raise Exception("Google AI response empty")
        return result_text
    
    def generate_realistic_sensor_data(self, scenario: str, time: str, weather: str) -> Dict:
        """
        Generate realistic sensor data based on context.
        
        Args:
            scenario: Environment scenario
            time: Time of day
            weather: Weather condition
            
        Returns:
            dict: Realistic sensor readings
        """
        # Temperature based on time and weather
        base_temp = {
            "dawn": 5, "morning": 12, "forenoon": 18, "noon": 25,
            "afternoon": 24, "evening": 20, "night": 15, "midnight": 8
        }[time]
        
        weather_temp_offset = {
            "clear": 3, "cloudy": 0, "rain": -5, "snow": -10,
            "fog": -2, "windy": -3, "storm": -8, "shower": -3
        }[weather]
        
        temperature = base_temp + weather_temp_offset + random.gauss(0, 2)
        
        # Humidity based on weather
        base_humidity = {
            "clear": 45, "cloudy": 65, "rain": 85, "snow": 70,
            "fog": 95, "windy": 50, "storm": 90, "shower": 80
        }[weather]
        
        humidity = max(20, min(100, base_humidity + random.gauss(0, 10)))
        
        wind_indices = self._sample_wind_direction_index(scenario)
        wind_direction = wind_indices

        if "climbing" in scenario or "exploration" in scenario:
            movement = "active"
        elif "walking" in scenario or "stroll" in scenario or "passage" in scenario:
            movement = "walking"
        else:
            movement = "walking"

        heading_deg = random.uniform(0.0, 360.0)
        heading_rad = math.radians(heading_deg)

        if movement == "active":
            base_speed = random.uniform(1.0, 2.5)
            noise = 0.2
        elif movement == "walking":
            base_speed = random.uniform(0.2, 0.8)
            noise = 0.08
        else:
            base_speed = random.uniform(0.05, 0.2)
            noise = 0.05

        imu = [
            base_speed * math.cos(heading_rad) + random.gauss(0.0, noise),
            base_speed * math.sin(heading_rad) + random.gauss(0.0, noise),
            9.8 + random.gauss(0.0, 0.2 if movement != "active" else 0.4),
            random.gauss(0.0, 0.15 if movement == "active" else 0.08),
            random.gauss(0.0, 0.15 if movement == "active" else 0.08),
            random.gauss(0.0, 0.06),
        ]

        return {
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "wind_direction": wind_direction,
            "imu": [round(x, 3) for x in imu],
            "movement_heading": round(heading_deg, 2),
            "movement_state": "active movement" if movement == "active" else "quiet movement",
            "movement": movement,
            "context": {
                "scenario": scenario,
                "time": time,
                "weather": weather
            }
        }
    
    def create_literary_prompt(
        self,
        sensor_data: Dict,
        style: str,
        required_keywords: Optional[List[str]] = None,
    ) -> str:
        """
        Create prompt for LLM to generate literary paragraph.
        
        Args:
            sensor_data: Sensor readings and context
            style: Literary style to use
            
        Returns:
            str: Formatted prompt for LLM
        """
        context = sensor_data["context"]
        temp = sensor_data["temperature"]
        humidity = sensor_data["humidity"]
        heading = sensor_data.get("movement_heading")
        wind_desc = self.wind_direction_to_description(
            sensor_data["wind_direction"],
            heading_degrees=heading,
        )
        movement_state = sensor_data.get("movement_state")
        movement_phrase = movement_state or ("active movement" if abs(sensor_data["imu"][0]) > 1 else "quiet movement")

        keyword_instructions = ""
        if required_keywords:
            keyword_list = ", ".join(required_keywords)
            keyword_instructions = (
                "\n**Required Keywords:**\n"
                f"- Include each of the following words at least once: {keyword_list}\n"
            )

        style_hint = self.style_guidelines.get(
            style,
            "Maintain the defining traits of the requested literary style throughout the narration.",
        )

        # Create detailed prompt in English
        prompt = f"""You are a novelist who writes with sensory and poetic style.
Please write a literary paragraph in {style} style based on the environmental sensor data collected by a robot narrator speaking in the first person.

**Context Information:**
- Location: {context['scenario'].replace('_', ' ')}
- Time: {context['time']}
- Weather: {context['weather']}

**Sensor Data:**
- Temperature: {temp}°C
- Humidity: {humidity}%  
- Wind direction: {wind_desc} (from robot's perspective)
- Movement: {movement_phrase}

**Style Guidance:**
- {style_hint}

**Writing Requirements:**
1. Write a 150-250 word paragraph
2. Use the first person (“I”, “my”) consistently; speak as the robot narrator
3. CRITICAL: NEVER write numerical values like "14.7°C" or "69.5%". The sensor data shows numbers for your reference only. You MUST translate them into sensory language.
   - ❌ WRONG: "14.7 degrees Celsius", "69.5% humidity"
   - ✅ RIGHT: "cool breath of air", "moisture-laden hush"
4. Use writing style that reflects {style} characteristics
5. Naturally incorporate feelings of temperature, humidity, and wind
6. Express interaction between the robot's movement and environment
{keyword_instructions}

Example beginnings: "He walked..." or "The air..." or "The wind..."

Literary paragraph:"""

        return prompt
    
    def wind_direction_to_description(
        self,
        direction_value: float,
        heading_degrees: Optional[float] = None,
    ) -> str:
        """Describe wind direction, optionally relative to movement heading."""

        wind_angle = self._wind_value_to_angle(direction_value)

        if heading_degrees is None:
            return self._absolute_wind_description(wind_angle)

        return self._relative_wind_description(wind_angle, heading_degrees)

    def _wind_value_to_angle(self, direction_value: float) -> float:
        """Convert stored wind value to degrees."""
        try:
            value = float(direction_value)
        except (TypeError, ValueError):
            return 0.0

        if 0 <= value < 16 and abs(value - round(value)) < 1e-6:
            return (round(value) % 16) * 22.5

        # assume value already in radians if within 0-2pi range
        if 0.0 <= value <= 2 * math.pi + 0.001:
            return math.degrees(value)

        # fallback: treat as degrees
        return value % 360

    def _absolute_wind_description(self, angle_degrees: float) -> str:
        directions = [
            (0, "blowing from the north"),
            (22.5, "coming from the north-northeast"),
            (45, "brushing from the northeast"),
            (67.5, "coming from the east-northeast"),
            (90, "blowing from the east"),
            (112.5, "coming from the east-southeast"),
            (135, "brushing from the southeast"),
            (157.5, "coming from the south-southeast"),
            (180, "pushing from the south"),
            (202.5, "coming from the south-southwest"),
            (225, "brushing from the southwest"),
            (247.5, "coming from the west-southwest"),
            (270, "blowing from the west"),
            (292.5, "coming from the west-northwest"),
            (315, "brushing from the northwest"),
            (337.5, "coming from the north-northwest"),
        ]

        angle = angle_degrees % 360
        for idx, (bound, phrase) in enumerate(directions):
            next_bound = directions[(idx + 1) % len(directions)][0]
            upper = next_bound if next_bound > bound else next_bound + 360
            if bound <= angle < upper:
                return phrase

        return "shifting winds"

    def _relative_wind_description(
        self,
        wind_angle: float,
        heading_degrees: float,
    ) -> str:
        """Describe wind direction relative to movement heading."""

        relative = ((wind_angle - heading_degrees + 180) % 360) - 180

        bands = [
            (-180, -157.5, "sweeping from the back-right"),
            (-157.5, -112.5, "brushing from the right rear"),
            (-112.5, -67.5, "brushing from the right"),
            (-67.5, -22.5, "curling from the front-right"),
            (-22.5, 22.5, "pressing from ahead"),
            (22.5, 67.5, "curling from the front-left"),
            (67.5, 112.5, "brushing from the left"),
            (112.5, 157.5, "brushing from the left rear"),
            (157.5, 180, "pushing from behind"),
        ]

        for lower, upper, phrase in bands:
            if lower <= relative < upper:
                return f"{phrase} (relative to motion)"

        return "shifting winds around the frame"

    def _sample_wind_direction_index(self, scenario: str) -> int:
        """Sample a discrete wind direction index (0-15) based on scenario."""
        scenario = (scenario or "").lower()

        if "indoor" in scenario:
            return 0
        if "beach" in scenario:
            candidates = [0, 1, 2, 15]  # favour onshore breezes around north/east
        elif "mountain" in scenario:
            candidates = list(range(16))
        elif "forest" in scenario or "field" in scenario:
            candidates = list(range(16))
        else:
            candidates = list(range(16))

        return random.choice(candidates)
    
    def load_novel_files(self) -> List[Dict[str, str]]:
        """Load novel files with their content, genre, and path."""
        if not self.input_dir:
            return []

        novel_entries: List[Dict[str, str]] = []
        file_patterns = ['*.txt', '*.md']

        for pattern in file_patterns:
            files = glob.glob(os.path.join(self.input_dir, '**', pattern), recursive=True)
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content) <= 100:  # Skip very short files
                            continue

                        novel_entries.append({
                            "content": content,
                            "path": file_path,
                            "genre": self._infer_genre_from_path(file_path)
                        })
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")

        return novel_entries

    def _infer_genre_from_path(self, file_path: str) -> str:
        """Infer novel genre from file path."""
        lowered = file_path.lower()
        if "modernist_novel" in lowered:
            return "modernist"
        if "travel_novel" in lowered:
            return "travel"
        return "unknown"
    
    def extract_text_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Extract meaningful text chunks from novel text.
        
        Args:
            text: Novel text content
            chunk_size: Target size for each chunk
            
        Returns:
            list: List of text chunks
        """
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed chunk_size, save current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 100]
        
        return chunks
    
    def analyze_text_for_environment(self, text: str) -> Dict:
        """
        Use LLM to analyze text and extract environmental information.
        
        Args:
            text: Text chunk to analyze
            
        Returns:
            dict: Environmental information extracted from text
        """
        prompt = f"""Analyze the following literary text and extract environmental information that could be measured by sensors. Focus on:

1. Temperature indicators (hot, cold, warm, cool, etc.)
2. Weather conditions (rain, snow, wind, fog, clear, etc.)
3. Time of day (morning, afternoon, evening, night, etc.)
4. Location type (city, forest, beach, indoor, etc.)
5. Movement/activity level (walking, running, still, etc.)

Text to analyze:
"{text}"

Please respond with a JSON object containing:
{{
  "temperature_hint": "cold/cool/mild/warm/hot",
  "weather": "clear/cloudy/rain/snow/fog/windy/storm",
  "time_of_day": "dawn/morning/forenoon/noon/afternoon/evening/night/midnight",
  "location": "city_walking/forest_exploration/beach_walking/indoor/etc",
  "movement": "still/walking/running/active",
  "confidence": 0.0-1.0
}}

Only extract information that is clearly indicated in the text. If uncertain, use "unknown" and lower confidence."""

        try:
            if self.use_google_ai:
                result_text = self._call_google_ai(
                    prompt="You are an expert at analyzing literary text for environmental details. Always respond with valid JSON.\n\n" + prompt,
                    temperature=0.0,
                    max_output_tokens=400
                )
            elif self.use_ollama and REQUESTS_AVAILABLE:
                response = self._call_ollama(
                    prompt="You are an expert at analyzing literary text for environmental details. Always respond with valid JSON.\n\n" + prompt,
                    temperature=0.3
                )
                result_text = response.strip()
            elif OPENAI_AVAILABLE and self.api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing literary text for environmental details. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                
                result_text = response.choices[0].message.content.strip()
            else:
                return self.simple_environmental_analysis(text)
            
            # Try to parse JSON response
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    # Fallback to simple analysis
                    return self.simple_environmental_analysis(text)
                    
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self.simple_environmental_analysis(text)
    
    def simple_environmental_analysis(self, text: str) -> Dict:
        """
        Fallback method for environmental analysis using simple patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Basic environmental information
        """
        text_lower = text.lower()
        
        # Temperature analysis
        if any(word in text_lower for word in ['hot', 'burning', 'sweltering', 'scorching']):
            temperature_hint = "hot"
        elif any(word in text_lower for word in ['warm', 'mild', 'pleasant']):
            temperature_hint = "warm"
        elif any(word in text_lower for word in ['cold', 'freezing', 'icy', 'chilly']):
            temperature_hint = "cold"
        elif any(word in text_lower for word in ['cool', 'crisp']):
            temperature_hint = "cool"
        else:
            temperature_hint = "mild"
            
        # Weather analysis
        weather = "clear"  # default
        if any(word in text_lower for word in ['rain', 'raining', 'shower', 'drizzle']):
            weather = "rain"
        elif any(word in text_lower for word in ['snow', 'snowing', 'blizzard']):
            weather = "snow"
        elif any(word in text_lower for word in ['fog', 'foggy', 'mist', 'misty']):
            weather = "fog"
        elif any(word in text_lower for word in ['wind', 'windy', 'breeze', 'gust']):
            weather = "windy"
        elif any(word in text_lower for word in ['storm', 'thunder', 'lightning']):
            weather = "storm"
        elif any(word in text_lower for word in ['cloud', 'cloudy', 'overcast']):
            weather = "cloudy"
            
        # Time analysis
        time_of_day = "afternoon"  # default
        if any(word in text_lower for word in ['morning', 'dawn', 'sunrise']):
            time_of_day = "morning"
        elif any(word in text_lower for word in ['noon', 'midday']):
            time_of_day = "noon"
        elif any(word in text_lower for word in ['evening', 'dusk', 'sunset']):
            time_of_day = "evening"
        elif any(word in text_lower for word in ['night', 'midnight', 'darkness']):
            time_of_day = "night"
            
        # Location analysis
        location = "city_walking"  # default
        if any(word in text_lower for word in ['forest', 'woods', 'trees', 'jungle']):
            location = "forest_exploration"
        elif any(word in text_lower for word in ['beach', 'ocean', 'sea', 'shore']):
            location = "beach_walking"
        elif any(word in text_lower for word in ['mountain', 'hill', 'peak', 'climb']):
            location = "mountain_climbing"
        elif any(word in text_lower for word in ['park', 'garden']):
            location = "park_stroll"
        elif any(word in text_lower for word in ['river', 'stream', 'creek']):
            location = "riverside_walking"
            
        # Movement analysis
        movement = "walking"  # default
        if any(word in text_lower for word in ['run', 'running', 'sprint', 'rush']):
            movement = "active"
        elif any(word in text_lower for word in ['still', 'motionless', 'stationary', 'sitting']):
            movement = "still"
        elif any(word in text_lower for word in ['walk', 'walking', 'stroll']):
            movement = "walking"
            
        return {
            "temperature_hint": temperature_hint,
            "weather": weather,
            "time_of_day": time_of_day,
            "location": location,
            "movement": movement,
            "confidence": 0.7  # Medium confidence for pattern-based analysis
        }
    
    def generate_sensor_from_environment(self, env_info: Dict) -> Dict:
        """
        Generate realistic sensor data based on environmental information.
        
        Args:
            env_info: Environmental information extracted from text
            
        Returns:
            dict: Generated sensor data
        """
        # Map temperature hints to ranges
        temp_ranges = {
            "hot": (25, 35),
            "warm": (18, 25),
            "mild": (12, 18),
            "cool": (5, 12),
            "cold": (-5, 5)
        }
        
        temp_range = temp_ranges.get(env_info.get("temperature_hint", "mild"), (12, 18))
        temperature = random.uniform(temp_range[0], temp_range[1])
        
        # Generate humidity based on weather
        humidity_base = {
            "clear": 45, "cloudy": 65, "rain": 85, "snow": 70,
            "fog": 95, "windy": 50, "storm": 90
        }
        base_humidity = humidity_base.get(env_info.get("weather", "clear"), 50)
        humidity = max(20, min(100, base_humidity + random.gauss(0, 10)))
        
        # Generate wind direction
        wind_direction = self._sample_wind_direction_index(env_info.get("location", ""))
        
        # Generate IMU based on movement
        movement = env_info.get("movement", "walking")
        if movement == "active":
            imu = [random.gauss(0, 2), random.gauss(0, 2), 9.8 + random.gauss(0, 0.5),
                   random.gauss(0, 0.3), random.gauss(0, 0.3), random.gauss(0, 0.1)]
        elif movement == "still":
            imu = [random.gauss(0, 0.1), random.gauss(0, 0.1), 9.8 + random.gauss(0, 0.05),
                   random.gauss(0, 0.02), random.gauss(0, 0.02), random.gauss(0, 0.01)]
        else:  # walking
            imu = [random.gauss(0, 0.5), random.gauss(0, 0.5), 9.8 + random.gauss(0, 0.2),
                   random.gauss(0, 0.1), random.gauss(0, 0.1), random.gauss(0, 0.05)]
        
        return {
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "wind_direction": wind_direction,
            "imu": [round(x, 3) for x in imu],
            "movement": movement,
            "context": {
                "scenario": env_info.get("location", "city_walking"),
                "time": env_info.get("time_of_day", "afternoon"),
                "weather": env_info.get("weather", "clear")
            }
        }
    
    def generate_literary_paragraph(self, prompt: str) -> Tuple[str, str, str]:
        """Generate literary paragraph and return text, method, model."""

        if self.use_google_ai:
            try:
                response = self._call_google_ai(
                    prompt="You are an excellent English literary writer.\n\n" + prompt,
                    temperature=0.8,
                    max_output_tokens=500
                )
                return response.strip(), "google_ai", self.google_model
            except Exception as e:
                print(f"Warning: Google AI call failed ({e}), falling back to alternative providers")

        if self.use_ollama and REQUESTS_AVAILABLE:
            try:
                response = self._call_ollama(
                    prompt="You are an excellent English literary writer.\n\n" + prompt,
                    temperature=0.8
                )
                return response.strip(), "ollama_llm", self.ollama_model
            except Exception as e:
                print(f"Warning: Ollama failed ({e}), falling back to template")

        if OPENAI_AVAILABLE and self.api_key:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an excellent English literary writer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.8
                )

                return response.choices[0].message.content.strip(), "openai_llm", "gpt-4"
            except Exception as e:
                print(f"Warning: OpenAI generation failed ({e}), using template")

        return self.generate_template_paragraph(prompt), "template_fallback", "template"

    def _normalize_environment_info(self, env_info: Dict, text: str) -> Dict:
        info = dict(env_info or {})
        text_lower = text.lower()

        time_patterns = {
            "morning": ["morning", "dawn", "sunrise", "breakfast", "daybreak", "first light"],
            "noon": ["noon", "midday", "meridian", "lunchtime"],
            "afternoon": ["afternoon", "midafternoon", "tea-time", "after lunch"],
            "evening": ["evening", "twilight", "sunset", "dusk"],
            "night": ["night", "midnight", "moonlit", "darkness"],
        }

        if info.get("time_of_day") in (None, "", "unknown"):
            for label, patterns in time_patterns.items():
                if any(pattern in text_lower for pattern in patterns):
                    info["time_of_day"] = label
                    break
        if info.get("time_of_day") in (None, "", "unknown"):
            info["time_of_day"] = random.choice(["morning", "afternoon", "evening"])

        location_patterns = {
            "indoor": ["indoors", "room", "hall", "chamber", "parlor", "kitchen", "library", "inside"],
            "forest_exploration": ["forest", "woods", "grove", "pines"],
            "field_crossing": ["field", "meadow", "pasture"],
            "beach_walking": ["beach", "shore", "sand", "coast"],
            "city_walking": ["street", "avenue", "city", "pavement", "alley", "market"]
        }

        if info.get("location") in (None, "", "unknown"):
            resolved = None
            for label, patterns in location_patterns.items():
                if any(pattern in text_lower for pattern in patterns):
                    resolved = label
                    break
            info["location"] = resolved or "city_walking"

        if info.get("location") == "indoor" and info.get("weather") in (None, "", "unknown"):
            info["weather"] = "clear"

        weather_patterns = {
            "rain": ["rain", "shower", "downpour", "drizzle"],
            "snow": ["snow", "blizzard", "flurry"],
            "fog": ["fog", "mist", "haze"],
            "windy": ["wind", "gust", "breeze", "gale"],
            "storm": ["storm", "thunder", "lightning"],
            "clear": ["clear", "bright", "sunny"]
        }

        if info.get("weather") in (None, "", "unknown"):
            resolved = None
            for label, patterns in weather_patterns.items():
                if any(pattern in text_lower for pattern in patterns):
                    resolved = label
                    break
            info["weather"] = resolved or "clear"

        if info.get("movement") in (None, "", "unknown"):
            for label, keywords in self.movement_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    if label == "active":
                        info["movement"] = "active"
                    elif label == "still":
                        info["movement"] = "still"
                    else:
                        info["movement"] = "walking"
                    break
        if info.get("movement") in (None, "", "unknown"):
            info["movement"] = "walking"

        if info.get("confidence") in (None, ""):
            info["confidence"] = env_info.get("confidence", 0.7)

        return info

    def _find_keyword(self, text: str, keywords: List[str]) -> Optional[str]:
        for word in keywords:
            if word in text:
                return word
        return None

    def _compute_alignment(self, paragraph: str, sensor_data: Dict) -> Tuple[Dict[str, str], float]:
        text_lower = paragraph.lower()
        mapping: Dict[str, str] = {}
        hits = 0
        total = 0

        temperature = sensor_data.get("temperature")
        if temperature is not None:
            total += 1
            if temperature <= 12:
                expected = "cold"
            elif temperature >= 22:
                expected = "warm"
            else:
                expected = "mild"
            match = self._find_keyword(text_lower, self.temperature_keywords[expected])
            conflict = None
            for label, words in self.temperature_keywords.items():
                if label == expected:
                    continue
                conflict = self._find_keyword(text_lower, words)
                if conflict:
                    break
            if match:
                hits += 1
                mapping["temperature"] = f"expected {expected}; matched '{match}'"
            elif conflict:
                mapping["temperature"] = f"expected {expected}; conflicting '{conflict}'"
            else:
                mapping["temperature"] = f"expected {expected}; no explicit cue"

        humidity = sensor_data.get("humidity")
        if humidity is not None:
            total += 1
            if humidity <= 40:
                expected = "dry"
                match = self._find_keyword(text_lower, self.humidity_keywords["dry"])
                conflict = self._find_keyword(text_lower, self.humidity_keywords["humid"])
            elif humidity >= 70:
                expected = "humid"
                match = self._find_keyword(text_lower, self.humidity_keywords["humid"])
                conflict = self._find_keyword(text_lower, self.humidity_keywords["dry"])
            else:
                expected = "neutral"
                match = conflict = None
            if expected == "neutral":
                mapping["humidity"] = "no strong expectation"
            elif match:
                hits += 1
                mapping["humidity"] = f"expected {expected}; matched '{match}'"
            elif conflict:
                mapping["humidity"] = f"expected {expected}; conflicting '{conflict}'"
            else:
                mapping["humidity"] = f"expected {expected}; no explicit cue"

        movement = sensor_data.get("movement")
        if movement:
            total += 1
            keywords = self.movement_keywords.get(movement, [])
            match = self._find_keyword(text_lower, keywords)
            if match:
                hits += 1
                mapping["movement"] = f"expected {movement}; matched '{match}'"
            else:
                mapping["movement"] = f"expected {movement}; no explicit cue"

        score = hits / total if total else 0.0
        return mapping, round(score, 3)

    def _is_low_quality_paragraph(
        self,
        paragraph: str,
        sensor_data: Dict,
        generation_method: str,
        alignment_score: float
    ) -> bool:
        text_lower = paragraph.lower()

        if len(paragraph.strip()) < 120:
            return True

        if re.search(r"\bgently wind\b", text_lower):
            return True

        temperature = sensor_data.get("temperature")
        if temperature is not None:
            if temperature <= 12 and self._find_keyword(text_lower, self.temperature_keywords["warm"]):
                return True
            if temperature >= 24 and self._find_keyword(text_lower, self.temperature_keywords["cold"]):
                return True

        humidity = sensor_data.get("humidity")
        if humidity is not None:
            if humidity <= 40 and self._find_keyword(text_lower, self.humidity_keywords["humid"]):
                return True
            if humidity >= 70 and self._find_keyword(text_lower, self.humidity_keywords["dry"]):
                return True

        if generation_method == "template_fallback":
            return True

        if alignment_score < 0.3:
            return True

        return False
    
    def generate_template_paragraph(self, prompt: str) -> str:
        """
        Fallback template-based paragraph generation.
        """
        templates = [
            "The air felt {temp_desc} as {wind_desc} wind brushed past his {body_part}. {movement_desc}, his footsteps pressed against the {ground_desc} ground, heading {destination_desc}.",
            
            "As the wind blew {wind_desc}, he felt a {skin_sensation}. In the {temp_desc} air, a {humidity_desc} atmosphere lingered, and he {movement_desc} continued walking.",
            
            "In the {time_desc}, {wind_desc} wind rustled his {clothing_desc}. As the {temp_desc} air enveloped his {face_desc}, his {step_desc} footsteps echoed on the {ground_desc} surface."
        ]
        
        # This is a simplified template - in practice, you'd extract context from prompt
        return random.choice(templates).format(
            temp_desc="cold" if "cold" in prompt else "warm",
            wind_desc="fiercely" if "strong" in prompt else "gently",
            body_part="face",
            movement_desc="carefully",
            ground_desc="firm",
            destination_desc="forward",
            skin_sensation="coolness",
            humidity_desc="moist",
            time_desc="early morning",
            clothing_desc="collar",
            face_desc="cheek",
            step_desc="slow"
        )
    
    def generate_novel_based_dataset(
        self,
        max_examples: int = 100,
        max_per_novel: int = None,
        max_files: int = None
    ) -> List[Dict]:
        """
        Generate dataset from real novel files by analyzing text for environmental information.
        
        Args:
            max_examples: Maximum number of examples to generate (global cap)
            max_per_novel: Deprecated; retained for backward compatibility (ignored)
            max_files: Optional cap on number of source novels to process
        
        Returns:
            list: Generated dataset examples
        """
        if not self.input_dir:
            print("Warning: No input directory specified. Falling back to synthetic generation.")
            return self.generate_dataset_batch(max_examples)
            
        print(f"Loading novel files from {self.input_dir}...")
        novel_entries = self.load_novel_files()

        if max_files is not None:
            novel_entries = novel_entries[:max_files]

        if max_per_novel is not None:
            print("Warning: max_per_novel is deprecated; processing will continue without per-file caps.")

        if not novel_entries:
            print("Warning: No novel files found. Falling back to synthetic generation.")
            return self.generate_dataset_batch(max_examples)
            
        print(f"Found {len(novel_entries)} novel files")

        dataset = []
        example_id = 0
        
        for text_idx, novel_entry in enumerate(novel_entries):
            novel_text = novel_entry["content"]
            source_genre = novel_entry.get("genre", "unknown")
            source_path = novel_entry.get("path")
            print(f"Processing novel {text_idx + 1}/{len(novel_entries)}...")
            
            # Extract chunks from this novel
            chunks = self.extract_text_chunks(novel_text)
            print(f"  Extracted {len(chunks)} text chunks")
            
            examples_from_this_novel = 0
            
            for chunk_idx, chunk in enumerate(chunks):
                if len(dataset) >= max_examples:
                    break
                
                try:
                    env_info_raw = self.analyze_text_for_environment(chunk)

                    if env_info_raw.get("confidence", 0) < 0.3:
                        continue  # Skip low-confidence extractions

                    env_info = self._normalize_environment_info(env_info_raw, chunk)

                    sensor_data = self.generate_sensor_from_environment(env_info)

                    inferred_style = self.infer_literary_style(chunk)
                    if source_genre == "modernist":
                        style = "modernist_novel"
                    elif source_genre == "travel":
                        style = "travel_essay"
                    else:
                        style = inferred_style

                    prompt = self.create_literary_prompt(sensor_data, style)

                    generation_method = "source_excerpt"
                    generation_model = "original_text"
                    alignment_mapping: Dict[str, str] = {}
                    alignment_score = 0.0
                    llm_success = False
                    paragraph_text = chunk[:500]
                    attempts_used = 0

                    if self.use_google_ai or self.use_ollama or self.api_key:
                        accepted = False
                        for attempt in range(1, self.max_generation_retries + 1):
                            paragraph_candidate, generation_method, generation_model = self.generate_literary_paragraph(prompt)
                            alignment_mapping, alignment_score = self._compute_alignment(paragraph_candidate, sensor_data)
                            if not self._is_low_quality_paragraph(paragraph_candidate, sensor_data, generation_method, alignment_score):
                                paragraph_text = paragraph_candidate
                                llm_success = generation_method != "template_fallback"
                                accepted = True
                                attempts_used = attempt
                                break
                            print(f"  Warning: Low quality paragraph detected (attempt {attempt})")
                        if not accepted:
                            print("  Skipping chunk due to persistent low-quality output")
                            continue
                    else:
                        alignment_mapping, alignment_score = self._compute_alignment(paragraph_text, sensor_data)
                        if self._is_low_quality_paragraph(paragraph_text, sensor_data, generation_method, alignment_score):
                            continue

                    entry = {
                        "id": f"novel_based_{example_id:06d}",
                        "sensor_data": sensor_data,
                        "literary_style": style,
                        "prompt": prompt,
                        "target_paragraph": paragraph_text,
                        "source_info": {
                            "novel_index": text_idx,
                            "chunk_index": chunk_idx,
                            "environmental_analysis": env_info_raw,
                            "environmental_analysis_normalized": env_info,
                            "source_genre": source_genre,
                            "source_path": source_path
                        },
                        "metadata": {
                            "generated_at": datetime.now().isoformat(),
                            "context": sensor_data["context"],
                            "source": "novel_analysis",
                            "style_category": style,
                            "generation_method": generation_method,
                            "generation_model": generation_model,
                            "llm_success": llm_success,
                            "alignment_score": alignment_score,
                            "sensor_to_text_mapping": alignment_mapping,
                            "generation_attempts": attempts_used if (self.use_google_ai or self.use_ollama or self.api_key) else 0
                        }
                    }

                    dataset.append(entry)
                    example_id += 1
                    examples_from_this_novel += 1

                    if (len(dataset)) % 10 == 0:
                        print(f"  Generated {len(dataset)}/{max_examples} examples")

                except Exception as e:
                    print(f"  Error processing chunk: {e}")
                    continue
            
            print(f"  Generated {examples_from_this_novel} examples from this novel")
            
            if len(dataset) >= max_examples:
                break
        
        print(f"Generated {len(dataset)} examples from novel analysis")
        print(f"Examples distributed across {len(set(ex['source_info']['novel_index'] for ex in dataset))} different novels")
        return dataset
    
    def infer_literary_style(self, text: str) -> str:
        """
        Infer the literary style of a text chunk.
        
        Args:
            text: Text to analyze
            
        Returns:
            str: Inferred literary style
        """
        text_lower = text.lower()
        
        # Simple heuristics for style detection
        if any(phrase in text_lower for phrase in ['he thought', 'she thought', 'consciousness', 'stream']):
            return "stream_of_consciousness"
        elif any(phrase in text_lower for phrase in ['saw', 'heard', 'felt', 'smelled', 'tasted']):
            return "sensory_description"
        elif any(phrase in text_lower for phrase in ['journey', 'traveled', 'destination', 'place']):
            return "travel_essay"
        elif len([s for s in text.split('.') if len(s.strip()) > 100]) > 2:
            return "modernist_novel"
        else:
            return "naturalist_style"

    def generate_dataset_batch(self, batch_size: int = 100) -> List[Dict]:
        """
        Generate a batch of sensor-literature pairs.
        
        Args:
            batch_size: Number of examples to generate
            
        Returns:
            list: Generated dataset examples
        """
        dataset = []
        generated = 0
        attempts = 0

        while generated < batch_size and attempts < batch_size * 3:
            attempts += 1

            scenario = random.choice(self.scenarios)
            time_ctx = random.choice(self.time_contexts)
            weather = random.choice(self.weather_contexts)
            style = random.choice(self.literary_styles)

            sensor_data = self.generate_realistic_sensor_data(scenario, time_ctx, weather)

            prompt = self.create_literary_prompt(sensor_data, style)

            paragraph_text = self.generate_template_paragraph(prompt)
            generation_method = "template_fallback"
            generation_model = "template"
            llm_success = False
            alignment_mapping, alignment_score = self._compute_alignment(paragraph_text, sensor_data)

            if self.use_google_ai or self.use_ollama or self.api_key:
                accepted = False
                for attempt in range(1, self.max_generation_retries + 1):
                    paragraph_candidate, generation_method, generation_model = self.generate_literary_paragraph(prompt)
                    alignment_mapping, alignment_score = self._compute_alignment(paragraph_candidate, sensor_data)
                    if not self._is_low_quality_paragraph(paragraph_candidate, sensor_data, generation_method, alignment_score):
                        paragraph_text = paragraph_candidate
                        llm_success = generation_method != "template_fallback"
                        accepted = True
                        break
                if not accepted:
                    continue
            else:
                if self._is_low_quality_paragraph(paragraph_text, sensor_data, generation_method, alignment_score):
                    continue

            entry = {
                "id": f"literary_{generated:06d}",
                "sensor_data": sensor_data,
                "literary_style": style,
                "prompt": prompt,
                "target_paragraph": paragraph_text,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "context": sensor_data["context"],
                    "source": "synthetic_batch",
                    "generation_method": generation_method,
                    "generation_model": generation_model,
                    "llm_success": llm_success,
                    "alignment_score": alignment_score,
                    "sensor_to_text_mapping": alignment_mapping
                }
            }

            dataset.append(entry)
            generated += 1

            if generated % 10 == 0:
                print(f"Generated {generated}/{batch_size} examples")

        return dataset
    
    def generate_dataset(self, num_examples: int = 100, max_files: int = None, output_path: str = None, use_novels: bool = True) -> List[Dict]:
        """
        Generate complete dataset with specified number of examples.
        
        Args:
            num_examples: Number of examples to generate
            max_files: Maximum number of files to use (for controlling file diversity)
            output_path: Optional path to save dataset
            use_novels: Whether to use novel analysis or synthetic generation
            
        Returns:
            list: Generated dataset
        """
        print(f"Generating {num_examples} examples...")
        
        if use_novels and self.input_dir:
            print("Using novel-based generation...")
            # Calculate max per novel to ensure diversity across novels
            dataset = self.generate_novel_based_dataset(
                max_examples=num_examples,
                max_per_novel=None,
                max_files=max_files
            )
        else:
            print("Using synthetic generation...")
            dataset = self.generate_dataset_batch(batch_size=num_examples)
        
        if output_path:
            self.save_dataset(dataset, output_path)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str):
        """Save dataset to JSON file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset saved to {filename}")
    
    def load_dataset(self, filename: str) -> List[Dict]:
        """Load dataset from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)


class DatasetAugmentor:
    """
    Augment existing dataset with variations and noise.
    """
    
    def __init__(self):
        pass
    
    def augment_sensor_data(self, sensor_data: Dict) -> List[Dict]:
        """
        Create variations of sensor data with realistic noise.
        
        Args:
            sensor_data: Original sensor readings
            
        Returns:
            list: Augmented sensor data variants
        """
        variations = []
        
        # Original data
        variations.append(sensor_data.copy())
        
        # Temperature variations
        for temp_delta in [-2, -1, 1, 2]:
            variant = sensor_data.copy()
            variant["temperature"] += temp_delta
            variations.append(variant)
        
        # Humidity variations  
        for humidity_delta in [-10, -5, 5, 10]:
            variant = sensor_data.copy()
            variant["humidity"] = max(0, min(100, variant["humidity"] + humidity_delta))
            variations.append(variant)
        
        # Wind direction variations (wrap around 0-15)
        for wind_delta in [-1, -2, 1, 2]:
            variant = sensor_data.copy()
            variant["wind_direction"] = (variant["wind_direction"] + wind_delta) % 16
            variations.append(variant)
        
        # IMU noise variations
        for noise_level in [0.1, 0.2]:
            variant = sensor_data.copy()
            variant["imu"] = [
                x + random.gauss(0, noise_level) for x in variant["imu"]
            ]
            variations.append(variant)
        
        return variations[:5]  # Return subset to avoid explosion


def main():
    """Example usage of the dataset generator."""
    
    # Initialize generator (without API key for template-based generation)
    generator = SyntheticLiteraryDatasetGenerator()
    
    # Generate small test dataset
    print("Generating synthetic literary dataset...")
    dataset = generator.generate_dataset_batch(batch_size=20)
    
    # Save dataset
    generator.save_dataset(dataset, "data/synthetic_literary_dataset.json")
    
    # Print example
    example = dataset[0]
    print("\n=== Example Generated Data ===")
    print(f"Scenario: {example['sensor_data']['context']}")
    print(f"Temperature: {example['sensor_data']['temperature']}°C")
    print(f"Humidity: {example['sensor_data']['humidity']}%")
    print(f"Wind Direction Index: {example['sensor_data']['wind_direction']}")
    print(f"Literary Style: {example['literary_style']}")
    print(f"Generated Paragraph:\n{example['target_paragraph']}")
    
    # Show augmentation example
    augmentor = DatasetAugmentor()
    augmented = augmentor.augment_sensor_data(example['sensor_data'])
    print(f"\nGenerated {len(augmented)} augmented variants")


if __name__ == "__main__":
    main()
