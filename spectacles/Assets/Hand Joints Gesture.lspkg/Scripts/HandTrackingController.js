// HandTrackingController.js
// Version 0.0.1
// Provides API to easilly access Hand Tracking status, 
// world positions of tracking points with ability to specify camera 
// Sends custom triggers on tracking started and lost

// @ui {"label":"Sends Behavior custom triggers:"}
// @ui {"label":"HAND_TRACKING_STARTED"}
// @ui {"label":"HAND_TRACKING_LOST"}
// @ui {"widget" : "separator"}
// @input bool advanced = false
// @ui {"widget":"group_start", "label" : "Object Tracking Components:", "showIf": "advanced"}
// @input Component.ObjectTracking Center
// @input Component.ObjectTracking Wrist
// @ui {"widget" : "separator"}
// @input Component.ObjectTracking Index0 
// @input Component.ObjectTracking Index1 
// @input Component.ObjectTracking Index2 
// @input Component.ObjectTracking Index3
// @ui {"widget" : "separator"}
// @input Component.ObjectTracking Middle0 
// @input Component.ObjectTracking Middle1 
// @input Component.ObjectTracking Middle2 
// @input Component.ObjectTracking Middle3
// @ui {"widget" : "separator"}
// @input Component.ObjectTracking Ring0
// @input Component.ObjectTracking Ring1 
// @input Component.ObjectTracking Ring2 
// @input Component.ObjectTracking Ring3 
// @ui {"widget" : "separator"}
// @input Component.ObjectTracking Pinky0 
// @input Component.ObjectTracking Pinky1 
// @input Component.ObjectTracking Pinky2  
// @input Component.ObjectTracking Pinky3
// @ui {"widget" : "separator"}
// @input Component.ObjectTracking Thumb0 
// @input Component.ObjectTracking Thumb1
// @input Component.ObjectTracking Thumb2
// @input Component.ObjectTracking Thumb3 


const onFoundTrigger = "HAND_TRACKING_STARTED";
const onLostTrigger = "HAND_TRACKING_LOST";

const CENTER = vec2.zero();

const MIN_LOST_FRAMES = 10;

// Hand Tracking 

// all tracking point names
const FULL_HAND_IDs = ["Center", "Wrist", "Index0", "Index1", "Index2", "Index3", "Middle0", "Middle1", "Middle2", "Middle3", "Ring0", "Ring1", "Ring2", "Ring3", "Pinky0", "Pinky1", "Pinky2", "Pinky3", "Thumb0", "Thumb1", "Thumb2", "Thumb3"];
// subset of points used to determine if hand is tracking or not
const MAIN_POINTS_IDs = ["Center", "Wrist"];



var HandTracking = function() {
    for (var i = 0; i < MAIN_POINTS_IDs.length; i++) {
        var m_id = MAIN_POINTS_IDs[i];
        if (!script[m_id]) {
            print("ERROR, " + m_id + " object tracking is not set, [" + m_id + "] Tracking Point was not created. Please check inputs under the advanced tab");
        }
    }

    for (var j = 0; j < FULL_HAND_IDs.length; j++) {
        var f_id = FULL_HAND_IDs[j];
        if (!script[f_id]) {
            print("WARNING, " + f_id + " object tracking is not set, [" + f_id + "] Tracking Point was not created. Please check inputs under the advanced tab");
        } else {
            this[f_id] = new HandPoint(f_id, script[f_id]);
        }
    }

    this.isTracking = function() {
        for (var i = 0; i < MAIN_POINTS_IDs.length; i++) {
            if (!this[MAIN_POINTS_IDs[i]] || !this[MAIN_POINTS_IDs[i]].isTracking()) {
                return false;
            }
        }
        return true;
    };
    //distance between hip and neck in local space
    this.getHandSize = function() {
        return this.Center.getLocalPosition().distance(this.Wrist.getLocalPosition());
    };
};

global.HandTracking = new HandTracking();

var TrackingState = { NONE: 0, TRACKING: 1 };
var stateMachine = new StateMachine();

stateMachine.addState(TrackingState.NONE, onNoneEnter, null, MIN_LOST_FRAMES);
stateMachine.addState(TrackingState.TRACKING, onTrackingEnter, onTrackingExit, 0);


script.createEvent("UpdateEvent").bind(function() {
    stateMachine.setState(getCurrentState());
});


function getCurrentState() {
    if (global.HandTracking && global.HandTracking.isTracking()) {
        return TrackingState.TRACKING;
    }
    return TrackingState.NONE;
}

//state change callbacks
function onTrackingEnter() {
    if (global.behaviorSystem) {
        global.behaviorSystem.sendCustomTrigger(onFoundTrigger);
    } else {
        print("WARNING, Please make sure behavior script exists in order to make custom triggers work");
    }
}

function onTrackingExit() {
    if (global.behaviorSystem) {
        global.behaviorSystem.sendCustomTrigger(onLostTrigger);
    } else {
        print("WARNING, Please make sure behavior script exists in order to make custom triggers work");
    }
}


function onNoneEnter() {
    print("Hands are not tracked.")
    
}


function HandPoint(id, trackingPoint) {
    this.name = id;
    this.objectTracking = trackingPoint;
    this.screenTransform = trackingPoint.getSceneObject().getFirstComponent("ScreenTransform");
}

HandPoint.prototype.isTracking = function() {
    return this.objectTracking.isTracking();
};

HandPoint.prototype.getWorldPosition = function() {
    return this.screenTransform.localPointToWorldPoint(CENTER);
};

HandPoint.prototype.getScreenPosition = function() {
    return this.screenTransform.localPointToScreenPoint(CENTER);
};

HandPoint.prototype.getLocalPosition = function() {
    return this.screenTransform.anchors.getCenter();
};


function StateMachine() {
    this.states = {};
    this.currentState = undefined;
    this.nextState = undefined;
    this.framesLeft = 0;

    this.addState = function(stateName, onStateEnter, onStateExit, minFrames) {
        this.states[stateName] = {
            onEnter: onStateEnter,
            onExit: onStateExit,
            minFrames: minFrames
        };
    };
    this.setState = function(stateName) {
        if (stateName == this.nextState) {
            this.framesLeft = this.framesLeft > 0 ? this.framesLeft - 1 : 0;
        } else {
            this.nextState = stateName;
            this.framesLeft = this.states[this.nextState].minFrames;
        }
        if (this.currentState != stateName && this.framesLeft == 0) {
            if (this.currentState != undefined && this.states[this.currentState].onExit) {
                this.states[this.currentState].onExit();
            }
            this.currentState = stateName;
            if (this.states[this.currentState].onEnter) {
                this.states[this.currentState].onEnter();
            }
            return;
        }
        
    };
    this.getState = function() {
        return this.currentState;
    };
}
