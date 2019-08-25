def ints(a,b):
    '''Generates the ints from a to b, inclusive and return a list of them'''
    ints  = list(range(a,b+1,1))
    return ints

#define a custom generator function
def char_range(c1,c2):
    '''Generates the characters from c1 to c1, inclusive'''
    for c in range(ord(c1), ord(c2)+1):
        yield chr(c)

def strings():
    '''Generates the characters from a to z, inclusive and return a list of them'''
    strings = []
    for c in char_range('a','z'):
        strings.append(c)
    return strings

def fixtures_gen():
    ints_here  = ints(-9999,9999)
    strings_here = strings()
    bools = [True,False]
    fixtures = [ints_here,strings_here,bools]
    return fixtures

def word_fixtures_gen(letters,uppercase):
    '''Generates 2,3,4 letter words into a list from a file in upper case or loswer case'''
    if uppercase == False:
        file_name = "{}_letter_words.txt".format(str(letters))
    else:
        file_name = "{}_letter_words_uppercase.txt".format(str(letters))
    with open(file_name) as f:
        lines = f.read().splitlines()
    return lines

def random_ints():
    import random
    random_ints_list = random.sample(range(100), 100)
    return random_ints_list