// @input bool isDefaultGesture = true
// @input string defaultGesture = "open" {"widget":"combobox", "values":[{"label":"Open Hand", "value":"open"}, {"label":"Close", "value":"close"}, {"label":"Horns", "value":"horns"}, {"label":"Index", "value":"index_finger"}, {"label":"Victory", "value":"victory"}], "showIf" : "isDefaultGesture"}
// @input string customGesture {"showIf" : "isDefaultGesture", "showIfValue" : "false"}
// @input float threshold = 0.35 {"widget":"slider", "min":0.0, "max":1.0, "step":0.01, "showIf" : "isDefaultGesture", "showIfValue" : "false"}
// @ui {"widget" : "separator"}
// @input string customResponseStart {"label" : "Response On Start", "hint" : "Custom Trigger for Behavior System"}
// @input string customResponseEnd{"label" : "Response On End", "hint" : "Custom Trigger for Behavior System"}
// @input bool debug = false

// @input bool advanced = false
// @input Component.ScriptComponent gestureController {"showIf" : "advanced"}

const COOL_DOWN = 0.1;
const THRESHOLD_MULTIPLIER = 0.1;
var sceneObject = script.getSceneObject();
var updateEvent = script.createEvent("UpdateEvent");
updateEvent.bind(checkTrigger);
updateEvent.enabled = false;

var triggerStarted = false;
var triggeredTime = 0;

if (script.isDefaultGesture) {
    if (script.gestureController) {
        script.gestureController.bindDefaultGesture(script.defaultGesture, script.customResponseStart, script.customResponseEnd, script.debug);
    }
} else {
    updateEvent.enabled = true;
}

checkInputs();

function checkTrigger() {
    if (!script.gestureController) {
        return;
    }
    
    var matching = script.gestureController.isMatchingCustomGesture(script.customGesture, script.threshold * THRESHOLD_MULTIPLIER);
    if (matching) {
        global.behaviorSystem.sendCustomTrigger(script.customResponseStart);
        triggerStarted = true;
        triggeredTime = getTime();
        printDebug(" started", script.debug);
    } else if (triggerStarted && getTime() - triggeredTime > COOL_DOWN) {
        global.behaviorSystem.sendCustomTrigger(script.customResponseEnd);
        triggerStarted = false;
        printDebug(" ended", script.debug);
    }
}

function checkInputs() {
    if (!script.gestureController) {
        printDebug("ERROR: Set GestureController script in Advanced section", true);
        return false;
    }
    
    return true;
}

function printDebug(message, force) {
    if (script.debug || force) {
        print("[Trigger " + sceneObject.name + "] " + message);
    }
}