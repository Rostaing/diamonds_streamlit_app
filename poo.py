# class Personnes:

#     def __init__(self, nom, prenom, age) -> None:
#         self._nom = nom
#         self.__prenom = prenom
#         self.age = age

#     def dire_bonjour(self, nom, prenom, age):
#         print(f"Bonjour {self.nom} {self.prenom}, vous avez {self.age}.")


# class Enfant(Personnes):
#     def __init__(self, nom, prenom, age, taille) -> None:
#         super().__init__(nom, prenom, age)
#         self.taile = taille


# p1 = Personnes("Davila", "Rostaing", 10)
# e1 = Enfant("Davila", "Rostaing", 10, 1.80)
# print(e1.nom)

# p1.dire_bonjour("Davila", "Rostaing", 10)
# print(p1._Personnes__prenom)
# print(p1._nom)
# print(x := 5)

import PyPDF2
import openai

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Fonction pour générer une réponse avec OpenAI
def generate_response(question, context):
    openai.api_key = 'YOUR_API_KEY'
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
        ]
    )
    return response['choices'][0]['message']['content']

# Exemple d'utilisation
pdf_text = extract_text_from_pdf('example.pdf')
response = generate_response("Quelle est la principale idée du document ?", pdf_text)
print(response)