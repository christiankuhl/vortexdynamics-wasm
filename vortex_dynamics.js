import init, { NVortexProblem } from './vortex_dynamics.js';

async function run(event) {
    const wasm = await init();
    try {
      var example = event.target.selectedOptions.item(0).vortices;
    } catch (TypeError) {
      var example = event.target.value;
    }
    var solution;
    var t;
    var tmax = 5;
    var canvas;
    var ctx;
    var vortices;
    var timer;
    // Hold canvas information
    var WIDTH;
    var HEIGHT;
    var RADIUS;
     // how often, in milliseconds, we check to see if a redraw is needed
    var INTERVAL = 20;
    var isDrag = false;
    var mx, my; // mouse coordinates
     // when set to true, the canvas will redraw everything
     // canvasValid = false; just sets this to false right now
     // we want to call canvasValid = false; whenever we make a change
    var canvasValid = false;
    // The node (if any) being selected.
    var mySel = null;
    var myHover = null;
    // we use a fake canvas to draw individual shapes for selection testing
    var ghostcanvas;
    var gctx; // fake canvas context
    // since we can drag from anywhere in a node
    // instead of just its x/y corner, we need to save
    // the offset of the mouse when we start dragging.
    var offsetx, offsety;
    // Padding and border style widths for mouse offsets
    var stylePaddingLeft, stylePaddingTop, styleBorderLeft, styleBorderTop;
    // Animation starting / stopping
    var paused = true;
    var just_started = true;
    var playbutton = document.querySelector(".button");
    var frame;
    // Context menu
    const menu = document.querySelector(".contextmenu");
    let menuVisible = false;

    document.getElementById("example").addEventListener("change", function(evt) { run(evt); });
    
    function Vortex() {
      this.x = 0;
      this.y = 0;
      this.gamma = 1;
    }

    Vortex.prototype = {
      draw: function(context) {
        if (context === gctx) {
          context.fillStyle = 'black'; // always want black for the ghost canvas
        } else {
          context.fillStyle = this.fill;
        }
        // We can skip the drawing of elements that have moved off the screen:
        if (this.x > WIDTH || this.y > HEIGHT) return;
        if (this.x + this.w < 0 || this.y + this.h < 0) return;
        // When selected (or hovered), the vortex receives an extra shadow, 
        // the corresponding Gamma control gets highlighted
        var ctrl = document.getElementById("gamma" + this.index);
        ctrl.classList.remove("gamma-highlight");
        if (mySel === this || myHover === this) {
          context.beginPath();
          var coords = toScreenCoordinates(this.x, this.y);
          context.arc(coords[0], coords[1], 4, 0, 2 * Math.PI, false);
          context.strokeStyle = "gray";
          context.lineWidth = 3;
          context.stroke();
          ctrl.classList.add("gamma-highlight")
        }
        // Vortex body
        context.beginPath();
        var coords = toScreenCoordinates(this.x, this.y);
        context.arc(coords[0], coords[1], 4, 0, 2 * Math.PI, false);
        context.fillStyle = heatMapColor(this.gamma);
        context.fill();
        context.lineWidth = 1;
        context.strokeStyle = 'black';
        context.stroke();
      }
    }

    function addVortex(x, y, gamma) {
      var vortex = new Vortex;
      vortex.x = x;
      vortex.y = y;
      vortex.gamma = gamma
      vortex.index = vortices.length;
      addGammaCtrl(vortex);
      vortices.push(vortex);
      canvasValid = false;
    }

    function removeVortex() {
      vortices.splice(mySel.index, 1);
      canvasValid = false;
      toggleMenu("hide");
      reorg();
    }

    function reorg() {
      const controls = document.getElementById("gamma-controls");
      while (controls.lastChild) {
        controls.lastChild.remove();
      }
      for (var i=0; i<vortices.length; i++) {
        var vortex = vortices[i];
        vortex.index = i;
        addGammaCtrl(vortex);
      }
    }

    function addGammaCtrl(vortex) {
      var div = document.createElement('div');
      div.class = "gamma-ctrl";
      div.id = "gamma" + vortex.index;
      var slider = document.createElement('input');
      slider.type = 'range';
      slider.class = 'slider';
      slider.min = -1;
      slider.max = 1;
      slider.step = 0.01;
      slider.value = vortex.gamma;
      slider.vortex = vortex
      var label = document.createElement('label');
      label.innerHTML = gammaExpr(slider);
      slider.label = label;
      slider.oninput = (e) => { e.target.vortex.gamma = e.target.value;
                                e.target.label.innerHTML = gammaExpr(e.target);
                                canvasValid = false;;
                              };
      div.appendChild(label);
      div.appendChild(slider);      
      document.getElementById("gamma-controls").appendChild(div);
    }

    function gammaExpr(slider) {
      var value = parseFloat(slider.value).toFixed(2);
      return `&#915;<sub>${slider.vortex.index + 1}</sub> = ${value}    `
    }

    function toMathCoordinates(x, y) {
      return [(x - WIDTH / 2) / RADIUS, (HEIGHT / 2 - y) / RADIUS]
    }

    function toScreenCoordinates(x, y) {
      return [WIDTH / 2 + x * RADIUS, HEIGHT / 2 - y * RADIUS]
    }

    function heatMapColor(value){
      var h = (1.0 - value) * 120
      return "hsl(" + h + ", 100%, 50%)";
    }

    function clear(c) {
      c.clearRect(0, 0, WIDTH, HEIGHT);
    }

    // Main draw loop.
    // While draw is called as often as the INTERVAL variable demands,
    // It only ever does something if the canvas gets invalidated by our code
    function mainDraw() {
      if (canvasValid) { return }
      clear(ctx);
      // draw domain boundary
      ctx.beginPath();
      ctx.arc(WIDTH / 2, HEIGHT / 2, RADIUS, 0, 2 * Math.PI, false);
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'black';
      ctx.stroke();
      // draw all vortices
      var l = vortices.length;
      for (var i = 0; i < l; i++) {
        vortices[i].draw(ctx);
      }
      canvasValid = true;
    }

    // Happens when the mouse is moving inside the canvas
    function onMouseMove(e){
      if (selectVortex(e)) { 
          canvasValid = false; 
          this.style.cursor = 'pointer';
      } else {
          this.style.cursor = 'auto';
      }
      if (isDrag) {
        getMouse(e);
        var math_coords = toMathCoordinates(mx - offsetx, my - offsety);
        mySel.x = math_coords[0];
        mySel.y = math_coords[1];
        canvasValid = false;
      }
    }

    function selectVortex(e) {
      getMouse(e);
      clear(gctx);
      var l = vortices.length;
      for (var i = l-1; i >= 0; i--) {
        // draw shape onto ghost context
        vortices[i].draw(gctx, 'black');
        // get image data at the mouse x,y pixel
        var imageData = gctx.getImageData(mx, my, 1, 1);
        // if the mouse pixel exists, select and break
        if (imageData.data[3] > 0) {
          myHover = vortices[i];
          var screen_coords = toScreenCoordinates(myHover.x, myHover.y);
          offsetx = mx - screen_coords[0];
          offsety = my - screen_coords[1];
          var math_coords = toMathCoordinates(mx - offsetx, my - offsety)
          myHover.x = math_coords[0];
          myHover.y = math_coords[1];
          canvasValid = false;;
          clear(gctx);
          return true;
        }
      }
      if (myHover != null) {
        canvasValid = false;
      }
      myHover = null;
      return false;
    }

    const toggleMenu = command => {
      menu.style.display = command === "show" ? "block" : "none";
      menuVisible = !menuVisible;
    };

    const setMenuPosition = ({ top, left }) => {
      menu.style.left = `${left}px`;
      menu.style.top = `${top}px`;
      toggleMenu('show');
    };

    // Happens when the mouse is clicked in the canvas
    function onMouseDown(e){
      if (e.button == 2) { return }
      if (menuVisible) { toggleMenu("hide") };
      if (selectVortex(e)) {
        isDrag = true;
        mySel = myHover;
        return
      };
      // haven't returned means we have selected nothing
      mySel = null;
      clear(gctx);
      canvasValid = false;;
    }

    function onMouseUp(){
      isDrag = false;
    }

    // adds a new vortex
    function onDoubleClick(e) {
      getMouse(e);
      var coords = toMathCoordinates(mx, my);
      addVortex(coords[0], coords[1], 1);
    }

    function onRightClick(e) {
      e.preventDefault();
      if (selectVortex(e)) {
        mySel = myHover;
        canvasValid = false;
      }
      if (mySel !== null) {
        const origin = {
          left: e.clientX,
          top: e.clientY
        };
        setMenuPosition(origin);
        return false;
      }
    }

    // Sets mx,my to the mouse position relative to the canvas
    // unfortunately this can be tricky, we have to worry about padding and borders
    function getMouse(e) {
          var element = canvas, offsetX = 0, offsetY = 0;
          if (element.offsetParent) {
            do {
              offsetX += element.offsetLeft;
              offsetY += element.offsetTop;
            } while ((element = element.offsetParent));
          }
          // Add padding and border style widths to offset
          offsetX += stylePaddingLeft;
          offsetY += stylePaddingTop;
          offsetX += styleBorderLeft;
          offsetY += styleBorderTop;
          mx = e.pageX - offsetX;
          my = e.pageY - offsetY
    }

    // initialize our canvas, add a ghost canvas, set draw loop
    // then add everything we want to intially exist on the canvas
    function initDisplay() {
      vortices = [];
      canvas = document.getElementById('vortex-canvas');
      HEIGHT = canvas.height;
      WIDTH = canvas.width;
      ctx = canvas.getContext('2d');
      RADIUS = Math.min(canvas.width, canvas.height) / 2 * .9;
      ghostcanvas = document.createElement('canvas');
      ghostcanvas.height = HEIGHT;
      ghostcanvas.width = WIDTH;
      gctx = ghostcanvas.getContext('2d');
      //fixes a problem where double clicking causes text to get selected on the canvas
      canvas.onselectstart = function () { return false; }
      // fixes mouse co-ordinate problems when there's a border or padding
      // see getMouse for more detail
      if (document.defaultView && document.defaultView.getComputedStyle) {
        stylePaddingLeft = parseInt(document.defaultView.getComputedStyle(canvas, null)['paddingLeft'], 10)     || 0;
        stylePaddingTop  = parseInt(document.defaultView.getComputedStyle(canvas, null)['paddingTop'], 10)      || 0;
        styleBorderLeft  = parseInt(document.defaultView.getComputedStyle(canvas, null)['borderLeftWidth'], 10) || 0;
        styleBorderTop   = parseInt(document.defaultView.getComputedStyle(canvas, null)['borderTopWidth'], 10)  || 0;
      }
      // set our events. Up and down are for dragging,
      // double click is for making new boxes
      canvas.onmousedown = onMouseDown;
      canvas.onmouseup = onMouseUp;
      canvas.ondblclick = onDoubleClick;
      canvas.onmousemove = onMouseMove;
      canvas.addEventListener("contextmenu", onRightClick);
      document.getElementById("delete-vortex").onclick = removeVortex;
      playbutton.onclick = toggleAnimation;
      
      // Add sample vortices
      loadExample(example);
    }

    function deleteVortices() {
      const controls = document.getElementById("gamma-controls");
      while (controls.lastChild) {
        controls.lastChild.remove();
      }
      vortices = [];
    }

    function loadExample(example) {
      console.log(example);
      deleteVortices();
      for (var i = 0; i < example.length; i++) {
        const vortex = example[i];
        addVortex(vortex.x, vortex.y, vortex.gamma);
      }
    }


    function toggleAnimation() {
        playbutton.classList.toggle("paused");
        paused = !paused;
        requestAnimationFrame(mainLoop);
        return false;
    }

    function done() {
      paused = true;
      playbutton.classList.toggle("paused");
      throw up;
    }

    function updateVortices() {
      for (let i = 0; i < vortices.length; i++) {
        let val = solution.next();
        if (val.done) { done() }
        vortices[i].x = val.value;            
      }
      for (let i = 0; i < vortices.length; i++) {
        let val = solution.next();
        if (val.done) { done() }
        vortices[i].y = val.value;
      }
    }

    function mainLoop() {
      if (paused) {
        timer = setInterval(mainDraw, INTERVAL);
        just_started = true;
      } else {
        clearInterval(timer);
        if (just_started) {
          var gamma = new Float64Array(vortices.length);
          var z = new Float64Array(2 * vortices.length);
          for (let i = 0; i < vortices.length; i++) {
            gamma[i] = vortices[i].gamma;
            z[i] = vortices[i].x;
            z[i + vortices.length] = vortices[i].y;
          }
          solution = NVortexProblem.new(gamma, z).mesh(tmax).values();    
          just_started = false;
        }
        try {
          updateVortices();
          canvasValid = false;
          mainDraw();
          requestAnimationFrame(mainLoop);
        } catch (up) {
          requestAnimationFrame(mainLoop);
        }
      }
    }
    
    initDisplay();
    mainLoop();
}

fetch("./examples.json").then(res => {
  res.json().then(examples => {
    var example_menu = document.example_menu.example;
    for (var i = 0; i < examples.length; i++) {
      var option = document.createElement("option");
      option.text = examples[i].name;
      option.vortices = examples[i].vortices;
      example_menu.add(option);
    }
    var example = example_menu.options[example_menu.selectedIndex].vortices;
    var event = {"target": {"value": example}};
    run(event);
  });
});
