// HandGesturesController.js
// Version: 0.0.1
// Event: Lens Initialized
// Description: The script that controls hand bones rotations and allows to compare current Gesture with the presets specified in Gesture Library
// Requires HandTrackingController.js

// @ui {"widget":"label", "label" : "Tap To Capture Gesture:"}
// @input bool tapGesture {"label" : "Enabled"}
// @ui {"widget": "separator", "showIf" : "tapGesture"}
// @ui {"widget":"label", "label":"To add custom gesture tap on the Preview Panel" , "showIf" : "tapGesture"}
// @ui {"widget":"label", "label":"A gesture output will appear in the Logger Panel", "showIf" : "tapGesture"}
// @ui {"widget":"label", "label":"Copy and paste new gesture to GestureController.js file", "showIf" : "tapGesture"}

const HAND_BONE_IDs = ["ThumbProximal", "ThumbIntermediate", "ThumbDistal", 
    "IndexProximal", "IndexIntermediate", "IndexDistal",
    "MiddleProximal", "MiddleIntermediate", "MiddleDistal",
    "RingProximal", "RingIntermediate", "RingDistal",
    "PinkyProximal", "PinkyIntermediate", "PinkyDistal"];
const VEC2_UP = vec2.up();
const PI = Math.PI;

var DEFAULT_GESTURES = {open : "", close: "", horns : "", index_finger : "", victory : "",
    open_end : "", close_end: "", horns_end : "", index_finger_end : "", victory_end : ""};

//----add custom hand gesture here----
//---remove the bones you don't need

var HandGesturesLibrary = {};

// To add custom pos:
//HandGesturesLibrary.MY_CUSTOM_GESTURE = Message copied from Logger after tapping screen

HandGesturesLibrary.THREE_FINGERS = {"IndexProximal":"3.079","IndexIntermediate":"0.080","IndexDistal":"0.096","MiddleProximal":"-3.132","MiddleIntermediate":"0.033","MiddleDistal":"-0.043","RingProximal":"3.129","RingIntermediate":"0.080","RingDistal":"-0.112","PinkyProximal":"0.301","PinkyIntermediate":"-0.133","PinkyDistal":"-3.083"};
HandGesturesLibrary.FOUR_FINGERS = {"ThumbProximal":"2.016","ThumbIntermediate":"0.465","ThumbDistal":"1.506","IndexProximal":"3.054","IndexIntermediate":"0.056","IndexDistal":"0.168","MiddleProximal":"3.109","MiddleIntermediate":"0.084","MiddleDistal":"0.024","RingProximal":"-2.879","RingIntermediate":"-0.153","RingDistal":"-0.163","PinkyProximal":"-3.018","PinkyIntermediate":"-0.233","PinkyDistal":"-0.096"};
HandGesturesLibrary.Tap_Gesture ={"ThumbProximal":"2.902","ThumbIntermediate":"0.145","ThumbDistal":"0.475","IndexProximal":"2.182","IndexIntermediate":"0.004","IndexDistal":"0.973","MiddleProximal":"2.476","MiddleIntermediate":"0.282","MiddleDistal":"0.618","RingProximal":"2.570","RingIntermediate":"0.194","RingDistal":"0.593","PinkyProximal":"2.812","PinkyIntermediate":"0.109","PinkyDistal":"0.400"}

//----do not edit below this line----

//create bones from TrackingPoints
// Define Bone function 
function Bone(id, start, end, parent) {
    this.name = id;
    this.parent = parent;
    this.start = start;
    this.end = end;
}

Bone.prototype.isTracking = function() {
    return this.start.isTracking() && this.end.isTracking();
};

Bone.prototype.getRotation = function() {
    var startPos = this.start.getWorldPosition();
    var endPos = this.end.getWorldPosition();
    var dir = new vec2(endPos.x, endPos.y).sub(new vec2(startPos.x, startPos.y)).normalize();
    var parentDir = VEC2_UP;
    if (this.parent) {
        var parentPos = this.parent.getWorldPosition();
        parentDir = new vec2(startPos.x, startPos.y).sub(new vec2(parentPos.x, parentPos.y)).normalize();
    }
    return Math.atan2(parentDir.x * dir.y - parentDir.y * dir.x, parentDir.dot(dir));
};

if (!global.HandTracking) {
    print("ERROR: Please make sure that HandTrackingController script exists and is higher in hierarchy");
    return;
}

// Create Finger Bones from TrackingPoints

var Bones = {};
var HT = global.HandTracking;

Bones.ThumbProximal = new Bone("ThumbProximal", HT.Thumb0, HT.Thumb1, HT.Wrist);
Bones.ThumbIntermediate = new Bone("ThumbIntermediate", HT.Thumb1, HT.Thumb2, HT.Thumb0);
Bones.ThumbDistal = new Bone("ThumbDistal", HT.Thumb2, HT.Thumb3, HT.Thumb0);

Bones.IndexProximal = new Bone("IndexProximal", HT.Index0, HT.Index1, HT.Wrist);
Bones.IndexIntermediate = new Bone("IndexIntermediate", HT.Index1, HT.Index2, HT.Index0);
Bones.IndexDistal = new Bone("IndexDistal", HT.Index2, HT.Index3, HT.Index0);

Bones.MiddleProximal = new Bone("MiddleProximal", HT.Middle0, HT.Middle1, HT.Wrist);
Bones.MiddleIntermediate = new Bone("MiddleIntermediate", HT.Middle1, HT.Middle2, HT.Middle0);
Bones.MiddleDistal = new Bone("MiddleDistal", HT.Middle2, HT.Middle3, HT.Middle0);

Bones.RingProximal = new Bone("RingProximal", HT.Ring0, HT.Ring1, HT.Wrist);
Bones.RingIntermediate = new Bone("RingIntermediate", HT.Ring1, HT.Ring2, HT.Ring0);
Bones.RingDistal = new Bone("RingDistal", HT.Ring2, HT.Ring3, HT.Ring0);

Bones.PinkyProximal = new Bone("PinkyProximal", HT.Pinky0, HT.Pinky1, HT.Wrist);
Bones.PinkyIntermediate = new Bone("PinkyIntermediate", HT.Pinky1, HT.Pinky2, HT.Pinky0);
Bones.PinkyDistal = new Bone("PinkyDistal", HT.Pinky2, HT.Pinky3, HT.Pinky0);


// returns current Gesture object
function getCurrentGesture() {
    var gesture = {};
    for (var i = 0; i < HAND_BONE_IDs.length; i++) {
        var id = HAND_BONE_IDs[i];
        var rotation = Bones[id].getRotation();
        gesture[id] = rotation.toFixed(3);
    }
    return gesture;
}

// prings current Gesture
function printGesture() {
    if (global.HandTracking.isTracking()) {
        print("Copy and paste this Gesture to GestureController.js file");
        print(JSON.stringify(getCurrentGesture()));
    } else {
        print("WARNING: Hand is not tracking, can't get current Gesture");
    }
}

//creates gesture object from the json string
function gestureFromJson(gestureJsonString) {
    var gesture = {};
    try {
        var parsedJson = gestureJsonString;
        for (var boneName in parsedJson) {
            gesture[boneName] = parseFloat(parsedJson[boneName]);
        }
        return gesture;
    } catch (e) {
        print("ERROR, " + e);
        return null;
    }
}

//checks whether current Hand Gesture is matching Control Gesture with given threshold
function isMatchingCustomGesture(gestureName, threshold) {
    if (!global.HandTracking || !global.HandTracking.isTracking()) {
        return false;
    }
    
    var gesture = HandGesturesLibrary[gestureName];
    var bonesCheckedCount = 0;
    var accumulatedAngleDiff = 0;
    
    for (var boneName in gesture) {
        if (Bones[boneName].isTracking()) {
            accumulatedAngleDiff += getDifference(gesture[boneName], Bones[boneName].getRotation()) / PI;           
        } else {
            return false;
        }
        
        bonesCheckedCount++;
    }
    return (accumulatedAngleDiff / bonesCheckedCount) < threshold;
}

function bindDefaultGesture(gestureName, customStartResponse, customEndResponse, debugMessage) {
    if (gestureName in DEFAULT_GESTURES) {
        DEFAULT_GESTURES[gestureName] = customStartResponse;
        if (customEndResponse) {
            DEFAULT_GESTURES[gestureName + "_end"] = customEndResponse;
        }
        if (global.HandTracking && global.HandTracking.Center) {
            global.HandTracking.Center.objectTracking.registerDescriptorStart(gestureName, function(gestureName) { 
                generateTriggerResponse(gestureName); if (debugMessage) {
                    print(gestureName + " started");
                }
            });
            global.HandTracking.Center.objectTracking.registerDescriptorEnd(gestureName, function(gestureName) { 
                generateTriggerResponse(gestureName + "_end"); if (debugMessage) {
                    print(gestureName + " ended");
                }
            });
        }
    } else {
        print(gestureName + " is not one of supported gestures");
    }
    
}


function generateTriggerResponse(gestureName) {
    if (global.behaviorSystem) {
        global.behaviorSystem.sendCustomTrigger(DEFAULT_GESTURES[gestureName]);
    }
}

function getDifference(x, y) {
    var a = x - y;
    a += (a > PI) ? -2 * PI : (a < -PI) ? 2 * PI : 0;
    return Math.abs(a);
}

//returns a list of Gestures (a Gesture) from the library by names
function getGesture(name) {
    var jsonString = HandGesturesLibrary[name];
    if (jsonString != null && jsonString != undefined) {
        return gestureFromJson(jsonString);
    } else {
        print("WARNING: " + name + " not found in the Gesture Library");
    }
}



if (script.tapGesture) {
    script.createEvent("TapEvent").bind(printGesture);
}

script.getGesture = getGesture;
script.isMatchingCustomGesture = isMatchingCustomGesture;
script.bindDefaultGesture = bindDefaultGesture;
