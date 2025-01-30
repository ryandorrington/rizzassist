#!/bin/bash

# Check if APP is set
if [ -z "$APP" ]; then
    echo "Error: APP environment variable not set"
    echo "Usage: APP=tinder ./rizzassist_automation.sh"
    echo "   or: APP=bumble ./rizzassist_automation.sh"
    exit 1
fi

# Set default ITERATIONS to 5 if not provided
ITERATIONS="${ITERATIONS:-2}"

# Check which app we're using and set appropriate values
if [ "$APP" = "tinder" ]; then
    ACTIONS=9
    ACTION_KEY=49  # Space bar
    SCREENSHOT_REGION="720,135,300,555"
    BASE_FOLDER="tinder_profiles"
elif [ "$APP" = "bumble" ]; then
    ACTIONS=6
    ACTION_KEY=125  # Down arrow
    SCREENSHOT_REGION="460,185,880,585"
    BASE_FOLDER="bumble_profiles"
else
    echo "Error: APP must be either 'tinder' or 'bumble'. Got: $APP"
    exit 1
fi

# Ensure the target base folder exists
if [ ! -d "$BASE_FOLDER" ]; then
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
  local region="$2"
  activate_browser
  screencapture -x -R "$region" "$output_path"
}

# Helper function to simulate key presses using AppleScript
# Key code references:
#  - 125 = Down arrow
#  - 124 = Right arrow
#  - 123 = Left arrow
#  - 49 = Space bar
press_key() {
  local key_code="$1"
  osascript -e "tell application \"System Events\" to key code $key_code"
  sleep 0.1  # Small delay after key press
}

SCREENSHOT_COUNT=1


# Main loop - repeat the entire process
for loop in $(seq 1 $ITERATIONS); do
  echo "Starting iteration $loop of $ITERATIONS..."
  
  # Create a new timestamped folder for this iteration
  TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
  TARGET_FOLDER="$BASE_FOLDER/$TIMESTAMP"
  mkdir -p "$TARGET_FOLDER" || {
    echo "Error: Unable to create target folder. Exiting."
    exit 1
  }
  echo "Created folder: $TARGET_FOLDER"
  
  # Create screenshots subdirectory
  SCREENSHOTS_DIR="$TARGET_FOLDER/screenshots"
  mkdir -p "$SCREENSHOTS_DIR" || {
    echo "Error: Unable to create screenshots directory. Exiting."
    exit 1
  }
  echo "Created screenshots directory: $SCREENSHOTS_DIR"
  
  # Take the initial screenshot
  SCREENSHOT_PATH="$SCREENSHOTS_DIR/screenshot_$SCREENSHOT_COUNT.png"
  take_screenshot "$SCREENSHOT_PATH" "$SCREENSHOT_REGION"
  ((SCREENSHOT_COUNT++))

  # Perform the actions
  for i in $(seq 1 $ACTIONS); do
    press_key $ACTION_KEY

    # Add a small random interval between 0.25 and 0.5 seconds
    SLEEP_DURATION=$(random_sleep)
    sleep "$SLEEP_DURATION"

    SCREENSHOT_PATH="$SCREENSHOTS_DIR/screenshot_$SCREENSHOT_COUNT.png"
    take_screenshot "$SCREENSHOT_PATH" "$SCREENSHOT_REGION"
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

  echo "Completed iteration $loop of $ITERATIONS"
  
  # Add a pause between iterations
  sleep 0.5
done

echo "Automation complete. All screenshots are in $SCREENSHOTS_DIR."
