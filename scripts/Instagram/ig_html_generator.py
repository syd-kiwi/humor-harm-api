# Topic keywords

topics = {
    "Health and Safety": [
        "COVID-19","vaccine","mask mandate","mental health","self-harm",
        "pandemic","health misinformation","lockdown","isolation","body image"
    ],
    "Politics and Society": [
        "US election","voting","political polarization","campaign","censorship",
        "freedom of speech","work from home","remote work burnout",
        "civil rights","government corruption"
    ],
    "Conflict and Global Events": [
        "Russia Ukraine","war","refugees","migration crisis","propaganda",
        "sanctions","natural disaster","climate refugee","peace negotiations","armed conflict"
    ],
    "Environmental and Ethical Issues": [
        "climate change","global warming","sustainability","animal rights","eco-activism",
        "wealth inequality","clean energy","deforestation","carbon footprint","environmental justice"
    ]
}

'''
    "Comedians": [
            "Ellen Degeneres", #258 million
            "Kevin Hart", #253 million
            "Joe Rogan", #62.8 million
            "Matt Rife", #33.8 million
            "Steve Harvey", #32.6 million
            "Nigel Ng", #26.5 million
            "Trevor Noah", #26.2 million
            "Russell Brand", #25.5 million
            "Adam Sandler", #25.1 million
            "Theo Von", #22.6 million
            "Gabriel Iglesias", #22.5 million
            "Jamie Foxx", #22.5 million
            "Trevor Wallace", #21.1 million
            "Ricky Gervais", #19.7 million
            "Amy Schumer", #17.5 million
            "Andrew Schulz", #16.5 million
            "Bill Maher", #15.6 million
            "Chelsea Handler", #14.4 million
            "Sarah Silverman", #13.9 million
            "Chris Rock", #13.7 million
        ]
    }

    topics = {
        "Funny Channels": [
        "Smosh", #24 million
        "David Dobrik", #13.7 million
        "CollegeHumor", #13 million
        "First We Feast", #6.94 million
        "Funny or Die", #3.26 million
        "The Tonight Show Starring Jimmy Fallon", #22 million
        "PewDiePie", #111 million
        "The Late Night Show", #0 million
        "Good Mythical Morning", #17.3 million
        "Tana Mongeau", #0 million
        "h3h3Productions", #6.32 million
        "Dolan Twins", #10.3 million
        "JennaMarbles", #20 million
        "Emma Chamberlain", #11.1 million
        "Markiplier", #36.6 million
        "Good Mythical MORE", #4.7 million
        "Jacksfilms", #4.7 million
        "Nigahiga", #21.4 million
        "Liza Koshy", #17.9 million
        "Shane Dawson TV", #19.9 million
        "The Try Guys", #8.6 million
        "Gus Johnson", #3.9 million
        "Cody Ko", #6.3 million
        "Danny Gonzalez", #7.7 million
        "Drew Gooden", #4 million
        "Kurtis Conner", #5.1 million
        "Penguinz0", #14.6 million
        "Ryan Trahan", #18.4 million
        "MrBeast", #238 million
        "CalebCity", #3.2 million
        "LongBeachGriffy", #4.1 million
        "King Bach", #2.6 million
        "NELK", #8.2 million
        "Trevor Wallace", #4.5 million
        "Mr Beast Gaming", #43.1 million
        "RDCworld1", #7.6 million
        "ImDontai", #2.9 million
        "Internet Historian", #4.8 million
        "TheOdd1sOut", #20 million
        "Jaiden Animations", #13.6 million
        "SMG4", #7.5 million
        "Brent Rivera", #25.9 million
        "Zach King", #17.7 million
        "Jordan Matter", #18.2 million
        "Lele Pons", #17.8 million
        "Kevin Langue", #3.92 million
        "Josh Joshson", #2.13 million
        "Don't Tell Comedy", #2.24 million
        "Netflix Is A Joke" #4.5 million
        "Comedy Central Stand-Up", #2.78 million
        "Jimmy O Yang" #2.04 million
        ]
    }
'''

from urllib.parse import quote_plus
import csv

BASE_URL = "https://www.instagram.com/explore/search/keyword/?q="

urls = {}

with open("ig_links.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "keyword", "url"])

    for category, words in topics.items():
        urls[category] = []

        for kw in words:
            # Example: append " funny" if you still want to bias toward humor
            query = f"{kw} funny"
            full_url = BASE_URL + quote_plus(query)

            urls[category].append(full_url)
            writer.writerow([category, kw, full_url])

# Print or use later in scraping loop
for cat, links in urls.items():
    print(f"\n[{cat}]")
    for link in links:
        print(link)
