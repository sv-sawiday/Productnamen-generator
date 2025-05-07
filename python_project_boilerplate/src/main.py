import logging
from pathlib import Path
import warnings

import pandas as pd
from openai import OpenAI

from config import API_KEY, DATA_PATH

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure Pandas to avoid silent downcasting warnings
pd.set_option('mode.chained_assignment', None)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def call_api(standard_dict, feature_dict):
    """
    Haalt de data van de externe API om de juiste features voor een product te kiezen. 

    Args: 
        None
    
    Returns:
        Features die met de naam samen gevoegd moeten worden
    """

    logging.info(f"API key loaded: {API_KEY[:4]}****")
    
    # de geoptimaliseerde prompt 
    PROMPT = f"""
    Je bent gespecialiseerd in het generen van productnamen voor badkamer producten. 
    De naam moet een duidelijke omschrijving geven van wat het product is en hoe het gebruikt kan worden, zonder het te technisch te maken.
    Een voorbeeld van een goede naam is (Fortifura Calvi Regendoucheset - thermostatisch - 25cm hoofddouche - staafhanddouche - Geborsteld koper PVD (Koper)).
    Hiervan moet jij zulke attributen uitkiezen thermostatisch - 25cm hoofddouche - staafhanddouche. De meest belangrijke eigenschappen dus!

    Kies de features uit de volgende dict, combineer de key en value in iets leesbaars: {feature_dict}. Gebruik GEEN afmetingen of arbitaire informatie!
    Het is ook van belang dat er geen verdubbelingen inzitten zoals: 2 Grepen, 2 functies en Knop of greep bediening. Kies dan de beste van de opties.
    De features die worden gekozen moeten wel volledig worden geschreven dus geen afkortingen.
    Geef wel voorkeur aan kortere en minder features.
    
    Als er afmetingen zijn alleen afmeting_commercieel (1876) nemen indien geen afmeting laat streepje weg, hetzelfde voor materiaalgroep (1967).
    Category1 moet als enkelvoudig product worden omschreven.
    Kleur moet op het einde.
    Deze worden genomen van {standard_dict}
    
    De output moet in de volgende format zijn: "brand - subbrand - catagory1 - afmeting_commercieel (1876) - feature_1 - feature_2 - etc. - materiaal - kleur (8)"
    """ 

    # creeeren van client met api key
    client = OpenAI(api_key = API_KEY)
    
    # set up van Chat Completions
    completion = client.chat.completions.create(
        model = "gpt-4.1-nano",  # test wat het optimale model is
        messages=[
            {
                "role": "system",
                "content": """
                Je bent een assistent voor een nederlands sanitairbedrijf en je spreekt alleen Nederlands. 
                Aan de hand van verschillende attributen die je aangeleverd krijgt is het jou taak om de best beschrijvende attributen te kiezen voor ieder product.
                Kies voor 0 tot 5 attributen afhankelijk wat het best het product omschrijft.
                """ # verander het aantal attributen indien nodig
            },
            { 
                "role": "user",
                "content": PROMPT
            }
        ],
        temperature = 0.1, # lage temperatuur voor hoge focus in het model zonder te veel creativiteit. De namen moeten namelijk allemaal gelijkwaardige sturcturen krijgen.
        top_p = 0.1, # neemt alleen de top 10% van de probability distributie voor de volgende token 
        max_completion_tokens = 2000 # gebaseerd op de intuitie dat deze taak niet te veel tokens nodig zal hebben
        )
    
    logging.info(f"Prompt send and data extracted")

    # het opvangen van de output data
    output = completion.choices[0].message.content

    return output


def load_data(): # hier moeten de paden nog van aan worden gepast in lijn met waar alles word geimporteerd
    """
    Laad de data in vanuit de data directory

    Args:
        input_pad (Path): locatie waar de input data opgeslagen staat. Dit is een excelsheet met daarin alle product features. 
    
    Returns:
        df met al de product features
    """

    base_dir = Path(__file__).resolve().parents[3]    
    data_dir = base_dir / "Data" / "Productnamen_generator" / "train_file.xlsx"
    
    df = pd.read_excel(data_dir, header=0)
    logging.info(f"Data loaded from path: {data_dir}")

    return df 


def get_features(row_data:pd.DataFrame):
    """
    Voor iedere rij met data worden de bruikbare features gefilterd en omgezet naar een dict

    Args:
        row_data (pd.DataFrame row with header): de data waarin de product informatie staat met daarbij de header om aan te duiden waar het over gaat.

    Returns:
        standard_dict (dict): de standaard informatie die in iedere naam moet zitten indien beschikbaar
        feature_dict (dict): de features die kunnen worden gekozen als aanvullende informatie voor de naam  
    """

    # verwijderen van de kolommen waar geen waardevolle data instaat
    row_data = row_data.dropna(axis=1)
    row_data = row_data.loc[:, (row_data != 0).any(axis=0)]

    # opsplitsen in standard- en feature-dictionaries
    standard_dict = {}
    feature_dict = {}

    # de standaard kolommen die in iedere naam zouden moeten zitten
    standard_columns = ["brand", "subbrand", "category1", "kleur (8)", "afmeting_commercieel (1876)", "materiaalgroep (1967)"]

    for col in standard_columns:
        if col in row_data:
            standard_dict[col] = row_data.iloc[0][col]

    # de andere kolommen zijn onderdeel van mogelijke features met enige uitzonderingen 
    feature_exceptions = ["brand", "subbrand", "category1", "kleur (8)", "afmeting_commercieel (1876)",
                          "kleurafwerking (736)", "kleur_reeks (753)", "first_activation_date (1820)",
                          "ten_behoeve_van (1995)", "marktdeelnemer (2053)", "film (17)", "garantie (1618)",
                          "materiaal (10)", "afmeting (1)", "lengte (2)", "breedte (4)", "diepte (7)",
                          "opties (15)", "breedte_reeks (20)", "bodemmaat (389)", "diameter_afvoergat (490)",
                          "lengte_reeks (1614)", "diepte_reeks (1750)"]

    for col in row_data:
        if col not in feature_exceptions:
            feature_dict[col] = row_data.iloc[0][col]

    return standard_dict, feature_dict


def create_names(df:pd.DataFrame):
    """
    selecteerd de features in de standaard naam conventie en zet deze in een standaard output

    Args:
        df (pd.DataFrame): import excelsheet met de productinformatie

    Returns:
        df (pd.Series): een lijst met alle juist geformatte namen met de best geselecteerde features 
    """
    
    name_series = pd.Series()

    for index in range(len(df[:10])):
        # ophalen van de rij
        row_data = df.iloc[[index]]

        # extraheren van de features
        standard_dict, feature_dict = get_features(row_data)
        logging.info(f"Feature dict len: {len(feature_dict)} and standard dict len: {len(standard_dict)} are created") 

        # geef beide is lijsten aan de api om de beste feature te extracten
        output = call_api(standard_dict, feature_dict)
        print(output)

        # voeg de nieuwe naam toe aan lijst met alle nieuwe namen
        output_series = pd.Series([output])
        name_series = pd.concat([name_series, output_series], ignore_index=True)
        print(name_series)

if __name__ == "__main__":
    df = load_data()
    create_names(df)

    # export moet nog toegevoegd worden
