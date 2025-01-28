#!/bin/bash
#
# rizzassist_automation.sh
#
# Creates a new folder inside profiles/ with a unique name based on
# the current date-time, takes screenshots, simulates key presses with random pauses,
# and finishes with a final left/right key press (90/10 distribution).
#

# 1. Ensure the target base folder exists (profiles/)
BASE_FOLDER="profiles"
if [ ! -d "$BASE_FOLDER" ]; then
  echo "Base folder not found. Creating $BASE_FOLDER..."
  mkdir -p "$BASE_FOLDER" || {
    echo "Error: Unable to create base folder. Exiting."
    exit 1
  }
fi

# Helper function to generate a random float between 0.25 and 0.5
random_sleep() {
  awk -v min=0.25 -v max=0.5 'BEGIN {
    srand(); 
    print min+rand()*(max-min)
  }'
}

# Helper function to activate Chrome/Safari before screenshots
activate_browser() {
  # Try Chrome first, then Safari if Chrome isn't running
  osascript -e '
    tell application "System Events"
      if exists process "Google Chrome" then
        tell application "Google Chrome" to activate
      else if exists process "Safari" then
        tell application "Safari" to activate
      end if
    end tell
    delay 0.1
  '
}

# Helper function for screenshots of specific region
take_screenshot() {
  local output_path="$1"
  activate_browser
  # Region parameters: x=460, y=185, width=880 (1340-460), height=585 (770-185)
  screencapture -x -R "460,185,880,585" "$output_path"
}

# Helper function to simulate key presses using AppleScript
# Key code references:
#  - 125 = Down arrow
#  - 124 = Right arrow
#  - 123 = Left arrow
press_key() {
  local key_code="$1"
  osascript -e "tell application \"System Events\" to key code $key_code"
  sleep 0.1  # Small delay after key press
}

SCREENSHOT_COUNT=1

# Main loop - repeat the entire process 10 times
for loop in {1..10}; do
  echo "Starting iteration $loop of 10..."
  
  # Create a new timestamped folder for this iteration
  TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
  TARGET_FOLDER="$BASE_FOLDER/$TIMESTAMP"
  mkdir -p "$TARGET_FOLDER" || {
    echo "Error: Unable to create target folder. Exiting."
    exit 1
  }
  echo "Created folder: $TARGET_FOLDER"
  
  # Take the initial screenshot after 2 second delay
  INITIAL_SCREENSHOT_PATH="$TARGET_FOLDER/screenshot_$SCREENSHOT_COUNT.png"
  echo "Waiting 2 seconds before taking initial screenshot..."
  sleep 2

  echo "Taking initial screenshot: $INITIAL_SCREENSHOT_PATH"
  take_screenshot "$INITIAL_SCREENSHOT_PATH"
  ((SCREENSHOT_COUNT++))

  # Press down key six times, each followed by a screenshot
  for i in {1..6}; do
    # Add a small random interval between 0.25 and 0.5 seconds
    SLEEP_DURATION=$(random_sleep)
    echo "Sleeping for $SLEEP_DURATION seconds before pressing down arrow."
    sleep "$SLEEP_DURATION"

    echo "Pressing down arrow (iteration $i)."
    press_key 125  # key code for Down arrow

    NEXT_SCREENSHOT_PATH="$TARGET_FOLDER/screenshot_$SCREENSHOT_COUNT.png"
    echo "Taking screenshot: $NEXT_SCREENSHOT_PATH"
    take_screenshot "$NEXT_SCREENSHOT_PATH"
    ((SCREENSHOT_COUNT++))
  done

  # Final action - press either right or left key
  #    90% chance to press right key, 10% chance for left key
  # RANDOM_CHOICE=$(( (RANDOM % 10) + 1 ))  # Generates a random number 1..10
  # if [ "$RANDOM_CHOICE" -le 9 ]; then
  #   echo "Final action: Pressing RIGHT arrow (90% chance)."
  #   press_key 124  # key code for Right arrow
  # else
  #   echo "Final action: Pressing LEFT arrow (10% chance)."
  #   press_key 123  # key code for Left arrow
  # fi

  echo "Final action: Pressing RIGHT arrow"
  press_key 124  # key code for Right arrow

  echo "Completed iteration $loop of 10"
  
  # Add a longer pause between iterations
  sleep 3
done

echo "Automation complete. All screenshots are in $TARGET_FOLDER."
