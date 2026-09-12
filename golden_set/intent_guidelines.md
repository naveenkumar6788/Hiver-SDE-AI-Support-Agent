# AppleSupport Intent Guidelines

## Purpose

These labels are used to classify incoming customer-support messages
for the AppleSupport AI support agent.

Each message should receive ONE primary intent.

---

## 1. battery_charging

Use when the main problem is related to:

- Battery draining quickly
- Poor battery life
- Battery percentage problems
- Battery health
- Phone not charging
- Charging problems
- Battery overheating or swelling

Examples:

- "My battery drains very quickly."
- "My iPhone won't charge."
- "Battery life became terrible after the update."

If the message mainly discusses an iOS update and only mentions battery
as a consequence, prefer `ios_software_update`.

---

## 2. ios_software_update

Use when the main problem is related to:

- iOS/macOS software updates
- Update installation failures
- Problems caused by an update
- Device freezing after an update
- Restarting after an update
- Wanting to downgrade/revert an update
- Software bugs

Examples:

- "My iPhone won't install the latest iOS."
- "My phone keeps restarting after the update."
- "iOS 11 broke my phone."

If the customer is specifically complaining about battery life without
focusing on the update itself, use `battery_charging`.

---

## 3. wifi_connectivity

Use when the main problem is related to:

- Wi-Fi connection
- Internet connection
- Network connection
- Wi-Fi disconnecting
- Incorrect Wi-Fi password problems
- Cellular/network connectivity when the issue is primarily connection-related

Examples:

- "My iPhone won't connect to Wi-Fi."
- "Wi-Fi keeps disconnecting."
- "Internet stopped working after the update."

---

## 4. app_problems

Use when a specific application is:

- Crashing
- Freezing
- Not opening
- Not responding
- Behaving incorrectly

Examples:

- "Netflix keeps crashing."
- "The app freezes every time I open it."
- "My keyboard app isn't working."

---

## 5. app_store_downloads

Use when the main problem involves:

- App Store
- Downloading apps
- Installing apps
- Updating apps through App Store
- App Store not loading

Examples:

- "I can't download any apps."
- "App Store keeps loading."
- "Why won't this app install?"

---

## 6. apple_music_itunes

Use when the main problem involves:

- Apple Music
- iTunes
- Music library
- Songs
- Playlists
- Music syncing
- Missing purchased music
- Music downloads

Examples:

- "My Apple Music won't play."
- "My iTunes library is missing songs."
- "Why can't I add songs to my playlist?"

---

## 7. apple_id_icloud

Use when the main problem involves:

- Apple ID
- iCloud
- Account login
- Password/account access
- Two-factor authentication
- iCloud synchronization
- iCloud storage
- Apple account settings

Examples:

- "I can't log into my Apple ID."
- "My iCloud isn't syncing."
- "I forgot my Apple ID password."

---

## 8. screen_display

Use when the main problem involves:

- Screen not responding
- Screen freezing
- Black screen
- Brightness
- Auto-brightness
- Display problems
- Lock-screen display problems
- Touch/display behavior

Examples:

- "My screen randomly goes black."
- "The screen doesn't respond to touch."
- "How do I turn off auto brightness?"

---

## 9. calls_cellular

Use when the main problem involves:

- Making calls
- Receiving calls
- Dropped calls
- Cellular service
- Phone signal
- Voicemail
- Call-related network problems

Examples:

- "I can't make phone calls."
- "My calls keep dropping."
- "I stopped receiving calls after the update."

---

## 10. audio_speaker

Use when the main problem involves:

- Speaker
- Sound
- Volume
- Microphone
- Audio output
- Static/noise
- AirPods audio

Examples:

- "My speaker doesn't work."
- "People can't hear me during calls."
- "My AirPods have a buzzing sound."

---

## 11. device_hardware

Use when the primary issue is a general device or hardware problem
that does not clearly belong to another intent.

Examples:

- Physical device malfunction
- SIM tray/device compatibility
- Camera hardware problem
- Hardware-specific failure
- Device won't power on when the cause is unclear

---

## 12. other_unclear

Use when:

- The message does not fit any defined intent
- The message is too vague
- The customer only expresses dissatisfaction
- There is insufficient information to determine the issue
- The message contains multiple unrelated issues and no clear primary issue

Examples:

- "This is ridiculous. Please help."
- "Why is this happening?"
- "Apple support is terrible."

---

# Labeling Rules

## Rule 1 — Choose the primary problem

If a message contains multiple problems, choose the problem that is
the main reason for contacting support.

Example:

"My battery is terrible since I installed iOS 11."

Primary intent:

`ios_software_update`

when the customer is clearly blaming the update.

---

## Rule 2 — Do not label based only on keywords

The presence of a word such as "app", "phone", "update", or "screen"
does not automatically determine the intent.

Read the entire message.

---

## Rule 3 — Prefer specific intents

If a message clearly belongs to a specific category, do not use
`other_unclear`.

---

## Rule 4 — Use other_unclear when uncertain

Do not force an uncertain message into an incorrect category.

---

## Rule 5 — One message = one primary intent

The classifier must ultimately predict one intent for each incoming
customer message.

---

## Rule 6 — Context can be used when available

If the incoming message is a follow-up such as:

"Yes, I already tried that."

the previous messages in the conversation may be needed to determine
the intent.

For standalone evaluation, however, the message should be understandable
enough to receive a reasonable label.

---

# Version

Initial taxonomy: v1

Number of intents: 12