from html4vision import Col, imagetable

# table description
cols = [
    Col('id1', 'ID'),
    Col('img', 'Label Map', '/Users/aachigorin/Downloads/Arina/*.jpg'),             # specify image content for column 2
    Col('img', 'Road Object Map', '/Users/aachigorin/Downloads/Arina/*.jpg'),     # specify image content for column 3
    Col('img', 'Amodel Road Mask', '/Users/aachigorin/Downloads/Arina/*.jpg'), # specify image content for column 4
]

# html table generation
imagetable(cols)