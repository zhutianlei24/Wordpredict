l1= [{'value': 'ID', 'type': 'Identifier'}, {'value': '(', 'type': 'Punctuator'}, {'value': '"STR"', 'type': 'String'}, {'value': ')', 'type': 'Punctuator'}, {'value': ';', 'type': 'Punctuator'}]
suffix = [{'value': '"STR"', 'type': 'String'}, {'value': ')', 'type': 'Punctuator'}]
for index, item in enumerate(l1):
    if item == suffix[0]:
        print(l1[:index ])