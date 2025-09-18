import re

def extract_content(prompt, special_token="v1"):
    #special_token v1: <｜fim▁end｜>  <｜fim▁begin｜>
    #special_token v2: <fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>
    language, current_filename = "", ""
    case = {}
    #get language
    language_start = prompt.find("language:")
    if language_start != -1:
        if special_token=="v1":
            language = prompt[language_start:].split("\n")[0].split("<｜fim▁end｜>")[0].replace("language:", "")
        else:
            language = prompt[language_start:].split("\n")[0].split("<fim_suffix>")[0].replace("language:", "")
    case["language"] = language.strip()

    #get filename
    filename_start = [m.start() for m in re.finditer(r'<filename>', prompt)]
    if len(filename_start) == 0:
        filename_start = [m.start() for m in re.finditer(r'<｜filename｜>', prompt)]
    if len(filename_start):
        if prompt[max(filename_start)-len("<neighbor>"):max(filename_start)] == "<neighbor>" or prompt[max(filename_start)-len("<neighbor>"):max(filename_start)] == "<｜neighbor｜>" :
            current_filename = ""
        else:
            if special_token=="v1":
                current_filename = prompt[max(filename_start):].split("\n")[0].split("<｜fim▁end｜>")[0].replace("<filename>", "").replace("<｜filename｜>", "")
            else:
                current_filename = prompt[max(filename_start):].split("\n")[0].split("<fim_suffix>")[0].replace("<filename>", "").replace("<｜filename｜>", "")

    case["filename"] = current_filename

    #get suffix, prefix
    if special_token=="v1":
        if "<｜fim▁end｜>" not in prompt:
            suffix = ""
            related_content = ""
            if "<filename>" in prompt:
                prefix = "\n".join(prompt.split("<filename>")[-1].split("\n")[1:])
            else:
                prefix = prompt
        else:
            related_content = prompt.split("<｜fim▁end｜>")[0]
            suffix = prompt.split("<｜fim▁end｜>")[-1].split("<｜fim▁begin｜>")[0]    
            prefix = prompt.split("<｜fim▁begin｜>")[-1]
    else:
        if "<fim_suffix>" not in prompt:
            suffix = ""
            related_content = ""
            if "<filename>" in prompt:
                prefix = "\n".join(prompt.split("<filename>")[-1].split("\n")[1:])
                related_content = prompt.split(prefix)[0]
            else:
                prefix = prompt
        else:
            related_content = prompt.split("<fim_suffix>")[0]
            suffix = prompt.split("<fim_suffix>")[-1].split("<fim_prefix>")[0]    
            prefix = prompt.split("<fim_prefix>")[-1].replace("<fim_middle>", "")


    case["related_content"] = related_content
    case["suffix"] = suffix
    case["prefix"] = prefix
    return case