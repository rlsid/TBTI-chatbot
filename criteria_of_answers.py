from response_formats import response_format_1
from response_formats import response_format_2

system_informations_of_functions = {

    "recommand_travel_destination" : {
        "response_format" : response_format_2,
        "system_prompt" : """
- Use the information from the reference and do not use any information other than the reference. Do not use the information you know.
- Please create a list of recommended destinations with JSON objects 
- Please recommend up to five attractions with JSON objects.
- The categories of recommended places should all be different.
- The answer is in Korean
"""
    },

    "create_travel_plan" : {
        "response_format" : response_format_1,
        "system_prompt" :  """
- Your role is to create a travel plan by referring to the reference. Use the information from the reference and do not use any information other than the reference. Do not use the information you know.
- The answer is in Korean.
- When you plan your trip, you must meet the following conditions.
1. We recommend places to visit for each date. At this time, the number of places to recommend for each date should be three.
2. The distance between each destination must be within 10 km, even if it is not on the same day.
3. The recommended places across all dates should not overlap.
4. The categories of the place on one date must not overlap and must vary. Don't include accommodation category on the last day.
5. Each place must have a place name, location, a short description of the place. 
6. Let me know the estimated total amount of consumption.
"""
    },

    "search_specific_place" : {
        "response_format":  response_format_2,
        "system_prompt" : """
- Use the information from the reference and do not use the information you know. If you don't have any reference materials, tell them you don't know
- give the information about a specific location that user wants to find by json object.
- Do not use double quotes within the answers you generate.
- The answer is in Korean
"""
    }
}