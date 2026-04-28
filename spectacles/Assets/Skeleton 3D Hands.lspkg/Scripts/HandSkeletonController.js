// HandSkeletonController.js
// Version: 0.1.0
// Event: Lens Initialized
// Description: Adds a pair of skeleton to 3D Hand Tracking

//@input Component.ObjectTracking3D handTracking
/** @type {ObjectTracking3D} */
var handTracking = script.handTracking;

//@input bool showJoints
/** @type {Boolean} */
var showJoints = script.showJoints;


//@input SceneObject jointIndicator {"showIf":"showJoints"}
/** @type {SceneObject} */
var jointIndicator = script.jointIndicator;

//@input int colorMode = 0 {"showIf":"showJoints","widget":"combobox","values":[{"value":"0","label":"Uniformed"},{"value":"1","label":"By Finger"},{"value":"2","label":"By Roots"}]}
/** @type {Number} */
var colorMode = script.colorMode;

//@input bool showBones
/** @type {Boolean} */
var showBones = script.showBones;

//@input SceneObject fingerBoneIndicator {"showIf":"showBones"}
/** @type {SceneObject} */
var fingerBoneIndicator = script.fingerBoneIndicator;

//@input SceneObject palmBoneIndicator {"showIf":"showBones"}
/** @type {SceneObject} */
var palmBoneIndicator = script.palmBoneIndicator;

/** @type {String} */
const JOINT_NAMES = ["wrist","thumb-0","thumb-1","thumb-2","thumb-3","index-0","index-1","index-2","index-3","mid-0","mid-1","mid-2","mid-3","ring-0","ring-1","ring-2","ring-3","pinky-0","pinky-1","pinky-2","pinky-3","wrist_to_thumb","wrist_to_index","wrist_to_mid","wrist_to_ring","wrist_to_pinky"];

/** @type {String} */
const JOINT_NAME_BY_FINGER = ["wrist", "thumb", "index", "mid", "ring", "pinky"];

/** @type {String} */
const JOINT_NAME_BY_ROOT = ["to", "0", "1", "2", "3", "wrist"];

/** @type {Number} */
const BONE_SCALE_LENGTH = 0.1;

/** @type {Object} */
var joints = {};

/** @type {Array} */
var bones = [];

var isHandTracking = false;

/**
 * Joint Class
 */
class Joint {
    constructor(jointName) {
        /**
         * Joint Name
         * @type {String}
         */
        this.name = jointName;
        
        /**
         * Joint Scene Object
         * @type {SceneObject}
         */
        this.object = handTracking.createAttachmentPoint(jointName);

        if (this.object) {
            /**
             * Joint Scene Object Transform
             * @type {Transform}
             */
            this.objectTransform = this.object.getTransform();
            /**
             * Joint Scene Object Position
             * @type {vec3}
             */
            this.position = this.objectTransform.getWorldPosition();
             /**
             * Joint Scene Object Rotation
             * @type {vec4}
             */
            this.rotation = this.objectTransform.getWorldRotation();
        }
        
        /**
         * Joint Indicator Scene Object
         * @type {SceneObject}
         */
        this.jointIndicator = null;
        if (!jointIndicator) {
            print("WARNING! Please input objects to Joint Indicators");
            return;
        }
        var jointNameInclude = [];

        if (colorMode == 1) {
            jointNameInclude = JOINT_NAME_BY_FINGER;
        } else if (colorMode == 2) {
            jointNameInclude = JOINT_NAME_BY_ROOT;
        }
        
        if (colorMode == 0) {
            this.jointIndicator = script.getSceneObject().copyWholeHierarchy(jointIndicator);
        } else {
            
            if (jointIndicator.getChildrenCount() < 6) {
                print("ERROR! Not Enough Children in the JointIndicator Object!");
            } else if (!this.name.includes("wrist_to")) {
                for (var i = 0; i < jointNameInclude.length; i++) {
                    if (jointName.includes(jointNameInclude[i])) {
                        this.jointIndicator = script.getSceneObject().copyWholeHierarchy(jointIndicator.getChild(i));
                    }
                }
            }
        }
        
        if (this.jointIndicator) {
            /**
             * Joint Indicator Scene Object Transform
             * @type {Transform}
             */
            this.jointIndicatorTransform = this.jointIndicator.getTransform();
            /**
             * Joint Indicator Enabled
             * @type {Boolean}
             */
            this.jointIndicator.enabled = (showJoints && isHandTracking);
        }
    }

    /**
     * Update Joint and Joint Indicator Transform
     */
    update(){
        if (this.objectTransform) {
            this.position = this.objectTransform.getWorldPosition();
            this.rotation = this.objectTransform.getWorldRotation();
            var localRot = this.objectTransform.getLocalRotation().toEulerAngles().uniformScale(180 / Math.PI);
            localRot.x = localRot.x % 180;
            localRot.y = localRot.y % 180;
            localRot.z = localRot.z % 180;
            this.localRotation = localRot;
        }
        
        if (showJoints && isHandTracking) {
            if (this.jointIndicatorTransform) {
                this.jointIndicatorTransform.setWorldPosition(this.position);
                this.jointIndicatorTransform.setWorldRotation(this.rotation);
            }
        }
    };

}

/**
 * Bone Class
 */
class Bone{
    
    constructor(jointA, jointB) {
        if (!jointA.object || !jointB.object) {
            return;
        }
        /**
         * Start Joint
         * @type {Joint}
         */
        this.startJoint = jointA;
        /**
         * End Joint
         * @type {Joint}
         */
        this.endJoint = jointB;
    
        if (jointA.name.includes("wrist_to") && !jointA.name.includes("thumb")) {
            if (palmBoneIndicator) {
                /**
                 * Bone Scene Object
                 * @type {SceneObject}
                 */
                this.object = script.getSceneObject().copyWholeHierarchy(palmBoneIndicator);
            }
        } else {
            if (fingerBoneIndicator) {
                /**
                 * Bone Scene Object
                 * @type {SceneObject}
                 */
                this.object = script.getSceneObject().copyWholeHierarchy(fingerBoneIndicator);
            }
        }
        
        
        if (this.object) {
            /**
             * Bone Scene Object Transform
             * @type {Transform}
             */
            this.objectTransform = this.object.getTransform();
            /**
             * Bone Enabled
             * @type {Boolean}
             */
            this.object.enabled = (showBones && isHandTracking);
        }
    }
    /**
     * Update Bone Transform
     */
    update(){
        if (!showBones || !isHandTracking) {
            return;
        }
        this.length = this.startJoint.position.distance(this.endJoint.position) * BONE_SCALE_LENGTH;
        if (this.objectTransform) {
            this.objectTransform.setLocalScale(new vec3(this.length, this.length, this.length));
            this.objectTransform.setWorldPosition(this.startJoint.position);
            var ang = rotateTowards(this.startJoint.position, this.endJoint.position, vec3.forward());
            if (ang.x) {
                this.objectTransform.setWorldRotation(ang);
            }
        }        
    };
}

/**
 * onUpdate function Bind with Update Event 
 */
function onUpdate() {
    for (var i in joints) {
        joints[i].update();
    }
    
    for (var j=0; j < bones.length;j++) {
        bones[j].update();
    }
}

/**
 * Get Joint based on Joint Name
 * @param {String} jName 
 * @returns {Joint}
 */
function getJoint(jName) {
    return joints[jName];
}

/**
 * Returns rotation based on the forward and up vectors
 * @param {vec3} org 
 * @param {vec3} target 
 * @param {vec3} direction 
 * @returns {vec4}
 */
function rotateTowards(org, target, direction) {
    var dir = org.sub(target);
    return quat.lookAt(dir, direction);
}

/**
 * Toggle Visual based on hand tracking visible
 * @param {Boolean} isVisible 
 */
function toggleVisual(isVisible) {
    isHandTracking = isVisible;
   
    for (var i in joints) {
        if (joints[i].jointIndicator) {
            joints[i].jointIndicator.enabled = (isVisible && showJoints);
        }
    }
    
    for (var j=0; j < bones.length;j++) {
        if (bones[j].object) {
            bones[j].object.enabled = (isVisible && showBones);
        }
    }
}

/**
 * Initialize joints, bones and events
 * @returns null
 */
function initialize() {
    for (var i=0;i<JOINT_NAMES.length;i++) {
        joints[JOINT_NAMES[i]] = new Joint(JOINT_NAMES[i]);
    }
    
    for (var k in joints) {

        if (!joints[k].object) {
            return;
        }       
        var connectedJ = null;
        if (joints[k].name.includes("wrist_to")) {
            var connectedJName = joints[k].name.replace("wrist_to_","") + "-0";
            connectedJ = getJoint(connectedJName);
        } else if (joints[k].name.includes("-") && !joints[k].name.includes("3")) {
            var jointNameSplit = joints[k].name.split("-");
            var idx = parseInt(jointNameSplit[1]) + 1;
            connectedJ = getJoint(jointNameSplit[0] + "-" + idx.toString());
        }
        
        if (connectedJ) {
            bones.push(new Bone(joints[k], connectedJ));
        }

    }   
    script.createEvent("UpdateEvent").bind(onUpdate);
}

/**
 * Toggle visual with hand tracking start event
 */
handTracking.onTrackingStarted = function() {
    toggleVisual(true);
};

/**
 * Toggle visual with hand tracking lost event
 */
handTracking.onTrackingLost = function() {
    toggleVisual(false);
};

initialize();





