// @input float holdDuration = 2.0
// @input string triggerName = "Recording_Start"

// @input SceneObject recordingTimer
// @input Component.Text timerText
// @input SceneObject handTriggerObject
// @input SceneObject redHand
// @input SceneObject greenHand
// @input Component.Text recordingCountdownText

var holdTimer = 0;
var isHolding = false;
var triggered = false;
var recordingTimeLeft = 0;
var recordingDuration = 10.0;

var onEnableEvent = script.createEvent("TurnOnEvent");
onEnableEvent.bind(function() {
    isHolding = false;
    holdTimer = 0;
    triggered = false;
    if (script.timerText) script.timerText.text = script.holdDuration.toFixed(1);
});

var updateEvent = script.createEvent("UpdateEvent");
updateEvent.bind(function(eventData) {
    var dt = getDeltaTime();

    if (isHolding && !triggered) {
        holdTimer += dt;
        if (script.timerText) script.timerText.text = (script.holdDuration - holdTimer).toFixed(1);

        if (holdTimer >= script.holdDuration) {
            triggered = true;
            global.behaviorSystem.sendCustomTrigger(script.triggerName);
            print("Recording triggered!");
        }
    }

    if (recordingActive) {
        recordingTimeLeft -= dt;
        if (script.recordingCountdownText) {
            if (recordingTimeLeft > 0) {
                script.recordingCountdownText.text = recordingTimeLeft.toFixed(1) + "s";
            } else {
                script.recordingCountdownText.text = "Processing...";
            }
        }
    }
});

var recordingActive = false;

// Call this when gesture starts
script.onGestureStart = function() {
    if (recordingActive) return;
    isHolding = true;
    holdTimer = 0;
    triggered = false;
}

// Call this when gesture ends
script.onGestureEnd = function() {
    isHolding = false;
    holdTimer = 0;
    triggered = false;
    print("Gesture hold broken");
}

// Expose functions for HandJointDataCollector to call
script.setRecordingActive = function(val) {
    recordingActive = val;
    if (script.handTriggerObject) script.handTriggerObject.enabled = !val;
    if (script.redHand) script.redHand.enabled = !val;
    if (script.greenHand) script.greenHand.enabled = !val;
    if (script.recordingTimer) script.recordingTimer.enabled = val;
    if (script.timerText) script.timerText.enabled = !val;
    if (script.recordingCountdownText) {
        script.recordingCountdownText.enabled = val;
        if (val) {
            recordingTimeLeft = recordingDuration;
            script.recordingCountdownText.text = recordingDuration.toFixed(1) + "s";
        } else {
            script.recordingCountdownText.text = "";
        }
    }
}
