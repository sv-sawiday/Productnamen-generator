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
    Je bent een assistent voor een nederlands sanitairbedrijf en je spreekt alleen Nederlands. 
    Je bent gespecialiseerd in het generen van productnamen voor badkamer producten. 
    De namen die je maakt zijn duidelijke omschrijvingen wat de unique selling point naar voren brengt op basis van de features. Leg vooral de focus op de producteigenschappen, laat algemene aspecten zoals service, prijs, eenheden, of niet unieke toepassingen achterwegen.

    Goede voorbeelden zijn als volgt:
        - Fortifura Calvi Regendoucheset - thermostatisch - 25cm hoofddouche - staafhanddouche - Geborsteld koper PVD (Koper)
        - Go by Van Marcke spiegelkast - 58x49.5x14.5cm - 3 deuren - kunststof - wit
        - Fortifura Calvi Fonteinkraan - 15.4cm - opbouw - 1gats - Geborsteld koper PVD (Koper)
        - ZEZA Pure half vrijstaand bad - 170x80x58cm - acryl - mat wit
        - Arcqua Havana Halfvrijstaand Bad - 170x80cm - links - mat groen
        - Adema Vygo meubelwastafel - 121x2x46cm - overloop - 2 wasbakken - 2 kraangaten - keramiek - wit
        - The Mosaic Factory Barcelona mozaïektegel - 30.9x30.9cm - wandtegel - vierkant - porselein - cream glans

    De standaard opbouw van de product naam is als volgt:
    "brand subbrand catagory1 - afmeting_commercieel (1) - feature_1 - feature_2 - etc. - materiaal - kleur (8)"

    brand subbrand catagory1, afmeting_commercieel (1), materiaal, en kleur (8) moeten er altijd in waar mogelijk. 

    De meeste features neem je uit de {standard_dict}, deze moeten er altijd inzitten. Wanneer er een mist laat je het streepje achterwegen. Ook moet voor category1 als een enkelvoudig product worden omschreven.

    De andere features worden gekozen uit de {feature_dict}. Zorg dat de focus van de features pas bij het type product uit de {standard_dict}. Kies voor 0 tot 5 MAXIMAAL features afhankelijk wat het best het product omschrijft. Geef de voorkeur aan kortere pakkende namen. Focus op de unique selling points, schrijf deze volledig uit zodat geen informatie verloren gaat. Zoals bij vernoeming formaat douchekop 25cm + het feit dat het een hoofddouche is. Voorkom dubbele benoeming van aspecten zoals met verlichting, binnen en/of onder verlichting en LED of 2 Grepen, 2 functies en Knop of greep bediening, kies dan de beste van de opties. 

    Geef als output steeds enkel de beste product naam in de aangegeven stijl. 
    """ 

    # creeeren van client met api key
    client = OpenAI(api_key = API_KEY)
    
    # set up van Chat Completions
    completion = client.chat.completions.create(
        model = "gpt-4.1-nano",  # test wat het optimale model is
        messages=[
            # {
            #     "role": "system",
            #     "content": """
            #     Je bent een assistent voor een nederlands sanitairbedrijf en je spreekt alleen Nederlands. 
            #     Aan de hand van verschillende attributen die je aangeleverd krijgt is het jou taak om de best beschrijvende attributen te kiezen voor ieder product.
               
            #     """ # verander het aantal attributen indien nodig
            # },
            { 
                "role": "system",
                "content": PROMPT
            }
        ],
        temperature = 0.15, # lage temperatuur voor hoge focus in het model zonder te veel creativiteit. De namen moeten namelijk allemaal gelijkwaardige sturcturen krijgen.
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
    standard_columns = ["brand", "subbrand", "category1", "kleur (8)", "afmeting (1)", "materiaalgroep (1967)"]

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

    for index in range(len(df)):
        # ophalen van de rij
        row_data = df.iloc[[index]]

        # extraheren van de features
        standard_dict, feature_dict = get_features(row_data)
        logging.info(f"Feature dict len: {len(feature_dict)} and standard dict len: {len(standard_dict)} are created") 

        # geef beide is lijsten aan de api om de beste feature te extracten
        output = call_api(standard_dict, feature_dict)

        # voeg de nieuwe naam toe aan lijst met alle nieuwe namen
        output_series = pd.Series([output])
        name_series = pd.concat([name_series, output_series], ignore_index=True)
        print(standard_dict)
        print(output)

    return name_series

if __name__ == "__main__":
    df = load_data()
    series = create_names(df)

    for name in series:
        print(name)

    # export moet nog toegevoegd worden
