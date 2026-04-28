// HandDirection.js
// Version: 0.1.0
// Event: On Awake
// Description: Detects the selected palm direction and triggers a custom event

//@input int palmDirection {"widget":"combobox", "values":[{"label":"Up", "value":0}, {"label":"Down", "value":1}, {"label":"Toward camera", "value":2}, {"label":"Away from camera", "value":3}]}
//@input float detectionPrecision = 0.6
//@input float lostPrecision = 0.3
//@ui {"widget": "separator"}
//@input bool useBehavior
//@input string onDirectionDetectedTrigger = "ON_DIRECTION_DETECTED" {"showIf" : "useBehavior"}
//@input string onDirectionLostTrigger = "ON_DIRECTION_LOST" {"showIf" : "useBehavior"}
//@ui {"widget":"separator"}
//@input bool callApiFunc
//@input Component.ScriptComponent scriptWithApi  {"showIf" : "callApiFunc"}
//@input string onDirectionDetectedFunction = "onDirectionDetected" {"showIf" : "callApiFunc"}
//@input string onDirectionLostFunction = "onDirectionLost" {"showIf" : "callApiFunc"}

var directionDetected;

function detectPalmDirection(hand) {
    if (hand === undefined) {
        directionDetected = false;
        return;
    }
    
    var handDirection = hand.isLeftHand ? hand.left : hand.right;  
    
    switch(script.palmDirection) {
        case 0: // up
            var referenceDirection = vec3.up();
            break;
        case 1: // down
            var referenceDirection = vec3.down();
            break;
        case 2: // toward camera
            var referenceDirection = vec3.forward();
            break;
        case 3: // away from camera
            var referenceDirection = vec3.back();
            break;
    }
    
    var dot = handDirection.dot(referenceDirection); 
    directionDetected = dot >= script.detectionPrecision;
    lostDetected = dot >= script.lostPrecision;
}

function onDirectionDetected(hand) {
    if (script.callApiFunc && script.scriptWithApi && script.onDirectionDetectedFunction && script.scriptWithApi[script.onDirectionDetectedFunction]) {
        script.scriptWithApi[script.onDirectionDetectedFunction](hand);
    }
    if (script.useBehavior && global.behaviorSystem && script.onDirectionDetectedTrigger) {
        global.behaviorSystem.sendCustomTrigger(script.onDirectionDetectedTrigger);
    }
}

function onDirectionLost() {
    if (script.callApiFunc && script.scriptWithApi && script.onDirectionLostFunction && script.scriptWithApi[script.onDirectionLostFunction]) {
        script.scriptWithApi[script.onDirectionLostFunction]();
    }
    if (script.useBehavior && global.behaviorSystem && script.onDirectionLostTrigger) {
        global.behaviorSystem.sendCustomTrigger(script.onDirectionLostTrigger);
    }
}

function onUpdate() {
    var hand = global.handTracking.getHand();  
    
    detectPalmDirection(hand);
  
    if (directionDetected) {
        onDirectionDetected();
    } else {
        onDirectionLost();
    }
}

function init() {
    directionDetected = false;
    script.createEvent("UpdateEvent").bind(onUpdate);    
}

init();