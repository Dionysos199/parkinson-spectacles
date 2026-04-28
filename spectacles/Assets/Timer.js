// @input float holdDuration = 2.0
// @input string triggerName = "Recording_Start"
// @input Component.Text timerText
var holdTimer = 0;
var isHolding = false;
var triggered = false;

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
        script.timerText.text = (script.holdDuration - holdTimer).toFixed(1);

        if (holdTimer >= script.holdDuration) {
            triggered = true;
            global.behaviorSystem.sendCustomTrigger(script.triggerName);
            print("Recording triggered!");
        }
    }
});

var recordingActive = false;
// Call this when gesture starts
script.onGestureStart = function() {
    if (recordingActive) return; // ignore gesture during recording
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
}