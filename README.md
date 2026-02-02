# Pursuit - Fursuit Character Recognition
Available on Telegram as [@furspybot](https://t.me/furspybot) and other names.

## Read [CLAUDE.md](CLAUDE.md) for a description of this project

## TODO (unordered):
01.02.2026

- [x] Keep metadata of which dataset and when the photo was ingested in the database
- [x] Create an ingestor to parse BARQ webapp fursuit pictures to a new dataset
- [x] Create a validation dataset from pictures that have not been downloaded from furtrack (we only download 2 out of many for each characters)
- [ ] Import an alias database from furtrack so that we can cross-validate characters appearing in e.g. nfc25 and furtrack
- [ ] Create other preprocessing pipelines and assess their score on the validation dataset, such as black-and-white preprocessing, brightness normalization, etc.
- [ ] Combine results of several detections / indices into a single weighted signal
- [ ] Prioritize most recently seen fursuits when scoring results
- [ ] Create nice icons for tg bots (I'm thinking of the fursuit with a labeled bounding box, so that it is obvious what this bot does)
- [ ] Make a feature to import the index data from another instance (to make it sync new fursuits across several running instances of the detector)
- [ ] User interface for submitting new pictures should be nice and fun to use
- [ ] Create an incentive for user data submission (game, find your fursuit parents, lookalikes, scores? etc) - make it beneficial or interesting to use
- [ ] Call to action to submit your own pictures to the database
- [ ] Find your own pictures in the database, even if they are not uploaded to furtrack?
- [ ] See other pictures of the same character from other databases, preferably not self-hosted for us to not get sued.
- [ ] Create an app that finds fursuit pictures in the camera roll (and maybe monitors it) and keeps a list of who you took pictures of
- [ ] Create call to action to upload some of the pictures you took to furtrack? Labels optional because I don't want to pollute furtrack with bad labels just yet. This will hurt training.
- [ ] Allow users to say "@bot this is character_name" and "@bot this not character_name"
- [ ] Try to make it possible to tag several people in the picture correctly (left to right?)
- [ ] Keep record of what each user was sending and doing with the image
- [ ] Use image infill to add occluded part of the fursuit head
- [ ] Add the full fursuit scanner to index on other parts of the body
- [ ] Use e.g. depth-anything to get extra angles on the fursuit
- [ ] Deprioritize low quality segments (low confidence, low res, unusual aspect crop for the prompt etc)
- [ ] Overlapping segments - can we resolve them? Esp. relevant for manual tagging from left to right
- [ ] Run clip on the fursuit crop and store that as a text search index
- [ ] Add text search mode (e.g. "neon green fox or dog with pink harness", "blue dragon with yellow horns and spikes on the head")

02.02.2026
- [ ] Add social login to the bot hello message, ie login with google/furaffinity/furtrack/barq/twitter etc so that we can analyze, attribute and upload images there later
- [ ] Add a link to the app to the bot hello message.
- [ ] Use Filesystem API in the webapp to list contents of users' folder periodically without having to select individual photos.
- [ ] Add a command to search for fursuits - input a name and get image URLs for a given character (or a partial name match). Make sure to include the origin in the response.
- [ ] Make the bot respond to edited messages
- [ ] Make the bot work in group chats with mentions and document how to set it up
- [ ] Find pictures with multiple fursuits and if we have at least 3, we can pick out which segment is the real tag by running self-similarity on a character pictures and mark all other segments as someone else. That way we reduce the noise.
- [ ] Run a self-similarity search on all database and cluster all potential segments to potentially assign it a tag, this is an extension of the previous point.
- [ ] Introduce a raw storage and process pipeline so that we can always recreate the results. This means a database/table with user edits and clustering results.
- [ ] Run a SAM2 + clip on each fragment instead of heavy SAM3 to determine if it's a fursuit head on the segment.
- [ ] Add mode to sync the index and database periodically to an upstream S3 bucket, not sure how to do that exactly but maybe shard them into append-only pieces (kinda like wal or wal itself) and upload those periodically, and merge on the client?
