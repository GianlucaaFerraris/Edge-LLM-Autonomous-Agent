import json
import random
from tqdm import tqdm
from openai import OpenAI # Usaremos la interfaz de OpenAI para conectar con Ollama o vLLM

# Configuración del "Modelo Maestro" (Local vía Ollama)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Tu lista extendida de categorías
categories = [
    # Cluster 1: Science & Tech
    "Software Engineering", "AI & Machine Learning", "Robotics", "Physics", 
    "Computer Architecture", "Space Exploration", "Renewable Energy", "Bioengineering", 
    "Cybersecurity", "Data Science", "Mathematics in Nature", "Materials Science", 
    "Internet of Things (IoT)", "Quantum Computing", "Telecommunications",
    # Cluster 2: Society & Future
    "AI Ethics", "Futurism", "Global Economics", "Climate Change", "Sociology", 
    "Urban Planning", "Philosophy of Technology", "Psychology of Social Media", 
    "Globalization", "The Future of Work", "Digital Rights", "Modern Education", 
    "Cryptocurrencies", "Space Law", "Transhumanism",
    # Cluster 3: Culture & Humanities
    "World History", "Movies & Series", "Literature", "Music Genres", "Art Movements", "Cultural Anthropology",
    "Fine Arts", "Mythology", "Archaeology", "Human Geography", "Linguistics", "Architecture History", "Religions of the World", "Philosophy", "Cultural Festivals",
    "Biology & Nature", "Environmental Science", "Zoology", "Botany", "Oceanography", "Meteorology", "Geology", "Ecology", "Astronomy", "Evolutionary Biology",
    # Cluster 4: Lifestyle & Experiences
    "Gastronomy", "Travel & Tourism", "Health & Fitness", "Sports", "Hobbies", 
    "Photography", "Gaming Culture", "Mindfulness & Mental Health", "Fashion & Identity", "Personal Finance",
    # Cluster 5: Personal & Local Context
    "Argentina (Culture & Society)", "University Challenges", "Career Development", "Family Dynamics", 
    "Social Etiquette", "Animal Behavior", "Nature & Gardening", "Public Speaking", "Volunteering & Social Impact", "Childhood Memories"
]

def get_topics_for_category(category, count=9):
    prompt = f"""Genera exactamente {count} temas de discusión cortos y conversacionales (NO preguntas) para practicar inglés técnico.
        Categoría: '{category}'.
        Estilo: Frases nominales o titulares cortos. Sin signos de interrogación. 
        Ejemplos: 
        - En lugar de '¿Cómo afecta la robótica al agro?', usa 'Posibles impactos de la robótica en el sector agronómico'.
        - En lugar de '¿Qué opinas de la ética en IA?', usa 'La importancia de la filosofía y ética en la Inteligencia Artificial'.
        - En lugar de '¿Cuáles son las comidas japonesas?', usa 'Gastronomía tradicional japonesa'.
        - En lugar de '¿Cuales son tus recuerdos de infancia que consideres propios de tu cultura?', usa 'Costumbres de tu infancia propias de tu nacion o cultura'.
        
        Devuelve ÚNICAMENTE un objeto JSON con una clave 'topics' que contenga una lista de strings en ESPAÑOL."""
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8 # Alta temperatura para mayor creatividad
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("topics", [])
    except Exception as e:
        print(f"\nError generating topics for {category}: {e}")
        return []

def main():
    all_topics = []
    print(f"Starting topic generation for {len(categories)} categories...")

    for cat in tqdm(categories):
        topics = get_topics_for_category(cat, count=9)
        all_topics.extend(topics)
        
    # Mezclamos y limitamos a 600
    import random
    random.shuffle(all_topics)
    final_topics = all_topics[:600]

    with open("topics.json", "w", encoding="utf-8") as f:
        json.dump(final_topics, f, ensure_ascii=False, indent=4)

    print(f"\nSuccessfully generated {len(final_topics)} topics in topics.json")

if __name__ == "__main__":
    main()