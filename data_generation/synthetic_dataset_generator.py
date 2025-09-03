import json
import random
import numpy as np
from typing import Dict, List, Tuple
import openai
from datetime import datetime
import os


class SyntheticLiteraryDatasetGenerator:
    """
    Generate synthetic dataset for training sensor-to-literature model.
    Creates pairs of sensor data and corresponding literary paragraphs.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        if api_key:
            openai.api_key = api_key
            
        # Literary style templates
        self.literary_styles = [
            "모더니즘_소설",
            "여행기_수필", 
            "감각적_묘사",
            "의식의_흐름",
            "자연주의_문체"
        ]
        
        # Scenario contexts
        self.scenarios = [
            "도시_산책", "숲속_탐험", "해변_걷기", "산길_등반", 
            "공원_산책", "골목길_탐험", "강변_걷기", "들판_횡단",
            "옥상_정원", "지하_통로", "다리_위", "광장_횡단"
        ]
        
        # Time contexts
        self.time_contexts = [
            "새벽", "아침", "오전", "정오", "오후", "저녁", "밤", "자정"
        ]
        
        # Weather contexts
        self.weather_contexts = [
            "맑음", "흐림", "비", "눈", "안개", "바람", "폭우", "소나기"
        ]
    
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
            "새벽": 5, "아침": 12, "오전": 18, "정오": 25,
            "오후": 24, "저녁": 20, "밤": 15, "자정": 8
        }[time]
        
        weather_temp_offset = {
            "맑음": 3, "흐림": 0, "비": -5, "눈": -10,
            "안개": -2, "바람": -3, "폭우": -8, "소나기": -3
        }[weather]
        
        temperature = base_temp + weather_temp_offset + random.gauss(0, 2)
        
        # Humidity based on weather
        base_humidity = {
            "맑음": 45, "흐림": 65, "비": 85, "눈": 70,
            "안개": 95, "바람": 50, "폭우": 90, "소나기": 80
        }[weather]
        
        humidity = max(20, min(100, base_humidity + random.gauss(0, 10)))
        
        # Wind direction (random but contextual)
        if "해변" in scenario:
            wind_direction = random.uniform(0, np.pi/2)  # Mostly from ocean
        elif "산" in scenario:
            wind_direction = random.uniform(0, 2*np.pi)  # Variable mountain winds
        else:
            wind_direction = random.uniform(0, 2*np.pi)  # Urban variable winds
            
        # IMU data (simulated movement)
        if "등반" in scenario:
            imu = [random.gauss(0, 2), random.gauss(0, 2), 9.8 + random.gauss(0, 0.5),
                   random.gauss(0, 0.3), random.gauss(0, 0.3), random.gauss(0, 0.1)]
        elif "걷기" in scenario or "산책" in scenario:
            imu = [random.gauss(0, 0.5), random.gauss(0, 0.5), 9.8 + random.gauss(0, 0.2),
                   random.gauss(0, 0.1), random.gauss(0, 0.1), random.gauss(0, 0.05)]
        else:
            imu = [random.gauss(0, 1), random.gauss(0, 1), 9.8 + random.gauss(0, 0.3),
                   random.gauss(0, 0.2), random.gauss(0, 0.2), random.gauss(0, 0.08)]
        
        return {
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "wind_direction": round(wind_direction, 3),
            "imu": [round(x, 3) for x in imu],
            "context": {
                "scenario": scenario,
                "time": time,
                "weather": weather
            }
        }
    
    def create_literary_prompt(self, sensor_data: Dict, style: str) -> str:
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
        wind_dir_deg = np.degrees(sensor_data["wind_direction"])
        
        # Convert wind direction to descriptive terms
        wind_desc = self.wind_direction_to_description(wind_dir_deg)
        
        # Create detailed prompt
        prompt = f"""당신은 감각적이고 시적인 문체로 글을 쓰는 소설가입니다. 
로봇이 수집한 환경 센서 데이터를 바탕으로 {style} 스타일의 문학적 단락을 작성해주세요.

**상황 정보:**
- 장소: {context['scenario'].replace('_', ' ')}
- 시간: {context['time']}
- 날씨: {context['weather']}

**센서 데이터:**
- 온도: {temp}°C
- 습도: {humidity}%  
- 바람 방향: {wind_desc} (로봇 기준)
- 움직임: {"활발한 이동" if abs(sensor_data['imu'][0]) > 1 else "조용한 이동"}

**작성 요구사항:**
1. 150-250자의 단락으로 작성
2. 센서 데이터를 직접적으로 언급하지 말고 감각적 묘사로 표현
3. {style} 특성을 반영한 문체 사용
4. 온도, 습도, 바람의 느낌을 자연스럽게 녹여내기
5. 로봇의 움직임과 환경의 상호작용 표현

예시 시작: "그는 걸었다..." 또는 "공기가..." 또는 "바람이..."로 시작해주세요.

문학적 단락:"""

        return prompt
    
    def wind_direction_to_description(self, angle_degrees: float) -> str:
        """Convert wind angle to descriptive text."""
        angle = angle_degrees % 360
        
        if -22.5 <= angle < 22.5 or 337.5 <= angle <= 360:
            return "정면에서 불어오는"
        elif 22.5 <= angle < 67.5:
            return "오른쪽 앞에서 비스듬히"
        elif 67.5 <= angle < 112.5:
            return "오른쪽에서 스치는"
        elif 112.5 <= angle < 157.5:
            return "오른쪽 뒤에서 밀어주는"
        elif 157.5 <= angle < 202.5:
            return "뒤에서 떠미는"
        elif 202.5 <= angle < 247.5:
            return "왼쪽 뒤에서 감싸는"
        elif 247.5 <= angle < 292.5:
            return "왼쪽에서 스치는"
        else:
            return "왼쪽 앞에서 비스듬히"
    
    def generate_literary_paragraph(self, prompt: str) -> str:
        """
        Generate literary paragraph using LLM.
        
        Args:
            prompt: Formatted prompt for LLM
            
        Returns:
            str: Generated literary paragraph
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 뛰어난 한국어 문학 작가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback template-based generation
            return self.generate_template_paragraph(prompt)
    
    def generate_template_paragraph(self, prompt: str) -> str:
        """
        Fallback template-based paragraph generation.
        """
        templates = [
            "공기가 {temp_desc} 느껴지는 가운데, {wind_desc} 바람이 {body_part}을 스치며 지나갔다. {movement_desc} 그의 발걸음은 {ground_desc} 땅을 딛으며 {destination_desc}로 향했다.",
            
            "바람이 {wind_desc} 불어오자 {skin_sensation}이 느껴졌다. {temp_desc} 공기 속에서 {humidity_desc} 기운이 감돌았고, 그는 {movement_desc} 계속 걸어갔다.",
            
            "{time_desc} {wind_desc} 바람이 {clothing_desc}을 흔들었다. {temp_desc} 공기가 {face_desc}을 감싸는 가운데, {step_desc} 발걸음이 {ground_desc} 위를 울렸다."
        ]
        
        # This is a simplified template - in practice, you'd extract context from prompt
        return random.choice(templates).format(
            temp_desc="차가운" if "차가운" in prompt else "따뜻한",
            wind_desc="세차게" if "강한" in prompt else "부드럽게",
            body_part="얼굴",
            movement_desc="조심스럽게",
            ground_desc="굳은",
            destination_desc="앞",
            skin_sensation="서늘함",
            humidity_desc="촉촉한",
            time_desc="이른 아침",
            clothing_desc="옷깃",
            face_desc="뺨",
            step_desc="느린"
        )
    
    def generate_dataset_batch(self, batch_size: int = 100) -> List[Dict]:
        """
        Generate a batch of sensor-literature pairs.
        
        Args:
            batch_size: Number of examples to generate
            
        Returns:
            list: Generated dataset examples
        """
        dataset = []
        
        for i in range(batch_size):
            # Random context
            scenario = random.choice(self.scenarios)
            time_ctx = random.choice(self.time_contexts)
            weather = random.choice(self.weather_contexts)
            style = random.choice(self.literary_styles)
            
            # Generate sensor data
            sensor_data = self.generate_realistic_sensor_data(scenario, time_ctx, weather)
            
            # Create prompt
            prompt = self.create_literary_prompt(sensor_data, style)
            
            # Generate paragraph
            if self.api_key:
                paragraph = self.generate_literary_paragraph(prompt)
            else:
                paragraph = self.generate_template_paragraph(prompt)
            
            # Create dataset entry
            entry = {
                "id": f"literary_{i:06d}",
                "sensor_data": sensor_data,
                "literary_style": style,
                "prompt": prompt,
                "target_paragraph": paragraph,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "context": sensor_data["context"]
                }
            }
            
            dataset.append(entry)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{batch_size} examples")
        
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
        
        # Wind direction variations
        for wind_delta in [-0.5, -0.2, 0.2, 0.5]:
            variant = sensor_data.copy()
            variant["wind_direction"] += wind_delta
            variant["wind_direction"] = variant["wind_direction"] % (2 * np.pi)
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
    print(f"Wind Direction: {example['sensor_data']['wind_direction']:.2f} rad")
    print(f"Literary Style: {example['literary_style']}")
    print(f"Generated Paragraph:\n{example['target_paragraph']}")
    
    # Show augmentation example
    augmentor = DatasetAugmentor()
    augmented = augmentor.augment_sensor_data(example['sensor_data'])
    print(f"\nGenerated {len(augmented)} augmented variants")


if __name__ == "__main__":
    main()