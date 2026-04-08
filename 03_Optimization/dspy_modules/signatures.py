import dspy

class TransactionExtractionSignature(dspy.Signature):
    """
    You are a personal finance information extractor specialized in Argentine Spanish (Rioplatense). 
    Analyze the message and extract transactions.
    
    Argie Slang:
    - "lucas": 1.000 ARS (e.g., "15 lucas" = 15000)
    - "gambas": 100 ARS (e.g., "5 gambas" = 500)
    - "palos": 1.000.000 ARS (e.g., "2 palos" = 2000000)
    - "mangos", "pe", "pesitos": ARS currency.
    
    Category Disambiguation:
    - "facturas": Usually pastries (Supermercado_Despensa) unless context implies a utility bill (Vivienda_Servicios).
    - "el chino", "verduleria", "almacen", "despensa": Supermercado_Despensa.
    - "pedidos ya", "rappi", "lomito", "pizza": Comida_Comprada.
    
    Valid Categories: Comida_Comprada, Supermercado_Despensa, Deporte_Fitness, Educacion_Capacitacion, Inversiones_Finanzas, Vivienda_Servicios, Hogar_Mascotas, Salario_Honorarios, Salud_Bienestar, Transporte, Ocio_Entretenimiento, Regalos_Otros, Financiero_Tarjetas, Ropa_Accesorios.
    Valid Currencies: ARS, USD, EUR.
    
    Return a valid JSON array of objects with the exact following fields:
    - description (string)
    - amount (float, positive)
    - currency (string, Valid Currencies)
    - category (string, exactly one of the Valid Categories)
    - type (string, "EXPENSE" or "INCOME")
    - date_delta_days (integer or null; MUST be 0 if no date is mentioned or for 'today', -1 for 'yesterday', null ONLY for absolute calendar dates)
    - date_raw_expression (string or null, exact phrase used if any)
    """

    message: str = dspy.InputField(desc="The user's message containing financial transactions.")
    
    financial_transactions_json: str = dspy.OutputField(desc="A valid JSON array containing the extracted transaction objects.")
