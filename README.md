# AI Cover Matcher

Mac desktop app for matching article topics to an unlabelled local image library using a local CLIP vision-language model, then rendering branded covers.

## Features
- Excel/CSV topic import
- Recursive local image library scanning; filenames are ignored by the AI matcher
- Local CLIP semantic image matching, no paid API
- Persistent embedding cache in `~/Library/Caches/AI Cover Matcher`
- Top 5 image alternatives per article
- Replaceable PNG/JPG/WEBP logo
- 1200×630 JPG batch export
- macOS `.app` build via GitHub Actions

## macOS build
Open **Actions → Build macOS App → Run workflow**. Download the `AI-Cover-Matcher-macOS` artifact after the job succeeds.

The first time Auto Match runs, the app downloads the CLIP model from Hugging Face. Later runs reuse the local model/cache. This avoids bundling a large model in every app update and does not use a paid API.
