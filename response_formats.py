response_format_1 = {
    "type": "json_schema",
    "json_schema" : {
        "name" : "A_general_answer",
        "schema" : {
            "type": "object",
            "properties" : {
                "answer": {
                    "type": "string",
                    "description" : "put your answer in the value. Escape to use quotation marks."
                },
                "place": {
                    "type": ["null", "object"],
                    "description": "This will be null."
                }
            },
            "required" : ["answer", "place"],
            "additionalProperties": False
        },
        "strict" : True
    }
}

response_format_2 = {
    "type": "json_schema",
    "json_schema" : {
        "name" : "A_unique_answer",
        "schema" : {
            "type": "object",
            "properties": {
                "answer" : {
                    "type": "string",
                    "description" : "put a short sentence that gives information about the places user wants to know. Do not use double quotes within this value"
                },
                "place": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties" : {
                            "place_name" : {
                                "type": "string",
                                "description" : "The name of the place"
                            },
                            "description" : {
                                "type": "string",
                                "description" : "A brief description of the place. make by using the keywords of the place"
                            },
                            "redirection_url": {
                                "type": "string",
                                "description" : "A URL for more information about the place"
                            }
                        },
                        "additionalProperties": False,
                        "required" : ["place_name", "description", "redirection_url"]
                    }
                }
            },
            "additionalProperties": False,
            "required" : ["answer", "place"] 
        },
        "strict" : True
    }
}