 #!/usr/bin/python
 # -*- coding: utf-8 -*-

import os
import codecs
mod_names = ['PARTICIPANTS', 'MODERATOR','MODERATORS','PANELISTS','HEMMER','MACCALLUM','TAPPER','BASH','HEWITT',
            'HARWOOD','QUICK','QUINTANILLA','REGAN','SEIB','SMITH','BLITZER','BAIER','KELLY','WALLACE','BAKER',
            'BARTIROMO','CAVUTO','MUIR','RADDATZ','GARRETT','STRASSEL','ARRARAS','DINAN','COOPER','LEMON',
            'LOPEZ','CORDES','COONEY','DICKERSON','OBRADOVICH','HOLT','MITCHELL','TODD','MADDOW','IFILL',
            'WOODRUFF','RAMOS','SALINAS','TUMULTY','LOUIS']
# Point the corpus root at the "Dem" directory, then the "rep" directory
corpus_root = '/Users/Goodgame/desktop/RedBlue/data/debate_data/rep/'
array = []

for file in os.listdir(corpus_root):
    array.append(file)

deleted = []
for doc in array:
    f = codecs.open(corpus_root + doc, 'r')
    first_word = []
    for line in f:
        line = line.replace(':',' ')
        first_word.append(line.split(None, 1)[0])
        print 'First word in document:'
        print first_word[0]
        if first_word[0] in mod_names:
            deleted.append(doc)
print 'Deleted file names:'
print deleted
print len(deleted)

f.close()

for file in deleted:
    os.remove(corpus_root + file)

# Dem starts with 4597 documents; ends with 3424

# Rep starts with 11546 documents; ends with 8997
