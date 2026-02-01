# Pursuit - Fursuit Character Recognition
Available on Telegram as [@furspybot](https://t.me/furspybot) and other names.

## Read [CLAUDE.md](CLAUDE.md) for a description of this project

## TODO (unordered):
01.02.2026

- Keep metadata of which dataset and when the photo was ingested in the database
- Create an ingestor to parse BARQ webapp fursuit pictures to a new dataset
- Create a validation dataset from pictures that have not been downloaded from furtrack (we only download 2 out of many for each characters)
- Import an alias database from furtrack so that we can cross-validate characters appearing in e.g. nfc25 and furtrack
- Create other preprocessing pipelines and assess their score on the validation dataset, such as black-and-white preprocessing, brightness normalization, etc.
- Combine results of several indices into a single weighted signal
- Prioritize most recently seen fursuits when scoring results
- Create nice icons for tg bots (I'm thinking of the fursuit with a labeled bounding box, so that it is obvious what this bot does)
- Make a feature to import the index data from another instance (to make it sync new fursuits across several running instances of the detector)
- User interface for submitting new pictures should be nice and fun to use
- Create an incentive for user data submission (game, find your fursuit parents, lookalikes, scores? etc) - make it beneficial or interesting to use
- Call to action to submit your own pictures to the database
- Find your own pictures in the database, even if they are not uploaded to furtrack?
- See other pictures of the same character from other databases, preferably not self-hosted for us to not get sued.
- Create an app that finds fursuit pictures in the camera roll (and maybe monitors it) and keeps a list of who you took pictures of
- Create call to action to upload some of the pictures you took to furtrack? Labels optional because I don't want to pollute furtrack with bad labels just yet. This will hurt training.

