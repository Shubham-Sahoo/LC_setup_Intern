function scatter(MAX_POINTS, size, texture = "") {
    let geometry = new THREE.BufferGeometry();
    let settings = {
        size: size,
        sizeAttenuation: false,
        alphaTest: 0.5,
        transparent: true,
        vertexColors: THREE.VertexColors
    };
    if (texture != "") {
        console.log(texture);
        settings["map"] = new THREE.TextureLoader().load(texture);
    }
    geometry.addAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    geometry.addAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    material = new THREE.PointsMaterial(settings);
    //     material.color.set(color);

    return new THREE.Points(geometry, material);
}

function scatterlcCloud(MAX_POINTS, size) {
    let geometry = new THREE.BufferGeometry();
    // let positions = [];
    // let colors = [];

    // for (var i = 0; i < points_arr.length / 4; ++i) {
    //     let x = points_arr[4 * i];
    //     let y = points_arr[4 * i + 1];
    //     let z = points_arr[4 * i + 2];
    //     let intensity = points_arr[4 * i + 3];

    //     if (enableInt16){
    //         x /= int16Factor;
    //         y /= int16Factor;
    //         z /= int16Factor;
    //         intensity /= int16Factor;   
    //     }
        
    //     positions.push(x, y, z);

    //     colors.push(0, intensity / 255, 0.2);
    // }

    // geometry.addAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    // geometry.addAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    geometry.addAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    geometry.addAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    geometry.computeBoundingSphere();

    let settings = {
        size: size,
        sizeAttenuation: false,
        alphaTest: 1.0,
        transparent: false,
        vertexColors: THREE.VertexColors
        // map: new THREE.TextureLoader().load("textures/sprites/disc.png")
    };

    let material = new THREE.PointsMaterial(settings);
    // let material = new THREE.PointsMaterial({size: 0.05, vertexColors: THREE.VertexColors});
    
    return new THREE.Points(geometry, material);
}

function scattersbCloud(MAX_POINTS) {
    let geometry = new THREE.BufferGeometry();
    let settings = {
        // size: 12,
        size: 5,
        sizeAttenuation: false,
        alphaTest: 1.0,
        transparent: false,
        color: new THREE.Color(1, 0, 0)
        // vertexColors: THREE.VertexColors
        // map: new THREE.TextureLoader().load("textures/sprites/disc.png")
    };

    geometry.addAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    geometry.computeBoundingSphere();
    material = new THREE.PointsMaterial(settings);
    material.color.set(new THREE.Color(0.4, 0, 0));

    return new THREE.Points(geometry, material);
}

function createHeatmapPlane(image_dataurl="") {
    // image_dataurl is a string of image bytes.
    var geometry = new THREE.PlaneGeometry( 70.4, 80, 1 );
    var material;
    material = new THREE.MeshBasicMaterial({ map: dummyTexture});
    if (image_dataurl == "") {
        // For a plane with some color.
        // material = new THREE.MeshBasicMaterial( { color: 0x011B39, side: THREE.SingleSide} );
        var dummyTexture = new THREE.DataTexture( new THREE.Color( 0xffffff ), 1, 1 );
        material = new THREE.MeshBasicMaterial({ map: dummyTexture });
    }
    else {
        var texture = new THREE.TextureLoader().load(image_dataurl);
        material = new THREE.MeshBasicMaterial({ map: texture });
    }
    var plane = new THREE.Mesh( geometry, material );
    plane.position.set(70.4 / 2, 0, -3);
    return plane;
}

// function heatmapParticles() {
//     var PARTICLE_SIZE = 0.1;
//     var geometry = new THREE.BufferGeometry();
//     var material = new THREE.PointsMaterial({
//         size: PARTICLE_SIZE,
//         vertexColors: THREE.VertexColors
//     });

//     var rowNumber = 100, columnNumber = 10000;
//     var xs = [], ys = [];
//     for (var i = 0; i < 176; i++) { xs.push(0.0 + i * (70.4 / 176)) }
//     for (var i = 0; i < 200; i++) { ys.push(-40 + i * (80.0 / 200)) }
//     var particleNumber = xs.length * ys.length;
//     var positions = new Float32Array(particleNumber * 3);
//     var colors = new Float32Array(particleNumber * 3);
//     for (x_index = 0; x_index < 176; ++x_index) {
//         for (y_index = 0; y_index < 200; ++y_index) {
//             var index = (x_index * 200 + y_index) * 3;

//             // put vertices on the XY plane
//             positions[index] = xs[x_index];
//             positions[index + 1] = ys[y_index];
//             positions[index + 2] = -3;

//             // just use random color for now
//             // colors[index] = Math.random();
//             // colors[index + 1] = Math.random();
//             // colors[index + 2] = Math.random();
//             colors[index] = colors[index + 1] = colors[index + 2] = 0.5;
//         }
//     }

//     // these attributes will be used to render the particles
//     geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
//     geometry.addAttribute('color', new THREE.BufferAttribute(colors, 3));
//     var particles = new THREE.Points(geometry, material);
//     return particles;
//     // scene.add(particles);
// }

function boxEdge(dims, pos, rots, edgewidth, color) {
    let boxes = [];
    for (var i = 0; i < dims.length; ++i) {
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        var edgeGeo = new THREE.EdgesGeometry(cube);
        let material = new THREE.LineBasicMaterial({
            color: color,
            linewidth: edgewidth
        });
        let edges = new THREE.LineSegments(edgeGeo, material);
        edges.position.set(pos[i][0], pos[i][1], pos[i][2]);
        edges.rotation.set(rots[i][0], rots[i][1], rots[i][2]);
        boxes.push(edges);
    }
    return boxes;
}

function boxEdgeWithLabel(dims, locs, rots, edgewidth, color, labels, lcolor) {
    let boxes = [];
    for (var i = 0; i < dims.length; ++i) {
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        var edgeGeo = new THREE.EdgesGeometry(cube);
        let material = new THREE.LineBasicMaterial({
            color: color,
            linewidth: edgewidth
        });
        let edges = new THREE.LineSegments(edgeGeo, material);
        edges.position.set(locs[i][0], locs[i][1], locs[i][2]);
        edges.rotation.set(rots[i][0], rots[i][1], rots[i][2]);

        var labelDiv = document.createElement( 'div' );
        labelDiv.className = 'label';
        labelDiv.textContent = labels[i];
        labelDiv.style.color = lcolor;
        // labelDiv.style.marginTop = '-1em';
        labelDiv.style.fontSize = "150%";
        // labelDiv.style.fontSize = "500%";
        var labelObj = new THREE.CSS2DObject( labelDiv );
        labelObj.position.set( 0, 0, 2 + dims[i][2]/2+locs[i][2] );
        edges.add(labelObj);
        boxes.push(edges);
    }
    return boxes;
}

function boxEdgeWithLabelV2(dims, locs, rots, edgewidth, color, labels, lcolor) {
    let boxes = [];
    for (var i = 0; i < dims.length; ++i) {
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        var edgeGeo = new THREE.EdgesGeometry(cube);
        let material = new THREE.LineBasicMaterial({
            color: color,
            linewidth: edgewidth
        });
        let edges = new THREE.LineSegments(edgeGeo, material);
        edges.position.set(locs[i][0], locs[i][1], locs[i][2]);
        edges.rotation.set(rots[i][0], rots[i][1], rots[i][2]);
        let labelObj = makeTextSprite(labels[i], {
            fontcolor: lcolor
        });
        labelObj.position.set(0, 0, dims[i][2] / 2);
        // labelObj.position.normalize();
        labelObj.scale.set(2, 1, 1.0);
        edges.add(labelObj);
        boxes.push(edges);
    }
    return boxes;
}

function box3D(dims, pos, rots, color, alpha) {
    let boxes = [];
    for (var i = 0; i < dims.length; ++i) {
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        let material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: alpha != 1.0,
            opacity: alpha
        });
        let box = new THREE.Mesh(cube, material);
        box.position.set(pos[i][0], pos[i][1], pos[i][2]);
        boxes.push(box);
    }
    return boxes;
}

function makeArrows(tails, heads) {
    var arrows = [];
    for (var i = 0; i < tails.length; i += 3) {
        var tail = new THREE.Vector3(tails[i], tails[i+1], tails[i+2]);
        var dir = new THREE.Vector3(heads[i]-tails[i],
                                    heads[i+1]-tails[i+1],
                                    heads[i+2]-tails[i+2]);
        var length = dir.length();
        dir.normalize();
        var hex = 0xffff00;
        var arrow = new THREE.ArrowHelper(dir, tail, length, hex);
        arrows.push(arrow);
    }
    return arrows;
}

function getKittiInfo(backend, root_path, info_path, callback) {
    backendurl = backend + '/api/readinfo';
    data = {};
    data["root_path"] = root_path;
    data["info_path"] = info_path;
    return $.ajax({
        url: backendurl,
        method: 'POST',
        contentType: "application/json",
        data: JSON.stringify(data),
        success: function (response) {
            return callback(response["results"][0]);
        }
    });
}

function loadKittiDets(backend, det_path, callback) {
    backendurl = backend + '/api/read_detection';
    data = {};
    data["det_path"] = det_path;
    return $.ajax({
        url: backendurl,
        method: 'POST',
        contentType: "application/json",
        data: JSON.stringify(data),
        success: function (response) {
            return callback(response["results"][0]);
        }
    });
}

function getPointCloud(backend, image_idx, with_det, callback) {
    backendurl = backend + '/api/get_pointcloud';
    data = {};
    data["image_idx"] = image_idx;
    data["with_det"] = with_det;
    return $.ajax({
        url: backendurl,
        method: 'POST',
        contentType: "application/json",
        data: JSON.stringify(data),
        success: function (response) {
            return callback(response["results"][0]);
        }
    });
}

function str2buffer(str) {
    var buf = new ArrayBuffer(str.length); // 2 bytes for each char
    var bufView = new Uint8Array(buf);
    for (var i = 0, strLen = str.length; i < strLen; i++) {
        bufView[i] = str.charCodeAt(i);
    }
    return buf;
}

function choose(choices) {
    var index = Math.floor(Math.random() * choices.length);
    return choices[index];
}

function makeTextSprite(message, opts) {
    var parameters = opts || {};
    var fontface = parameters.fontface || 'Helvetica';
    var fontsize = parameters.fontsize || 70;
    var fontcolor = parameters.fontcolor || 'rgba(0, 1, 0, 1.0)';
    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');
    context.font = fontsize + "px " + fontface;
  
    // get size data (height depends only on font size)
    var metrics = context.measureText(message);
    var textWidth = metrics.width;
  
    // text color
    context.fillStyle = fontcolor;
    context.fillText(message, 0, fontsize);
  
    // canvas contents will be used for a texture
    var texture = new THREE.Texture(canvas)
    texture.minFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
  
    var spriteMaterial = new THREE.SpriteMaterial({
        map: texture,
    });
    var sprite = new THREE.Sprite(spriteMaterial);
    // sprite.scale.set(5, 5, 1.0);
    return sprite;
  }
  