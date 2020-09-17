import init, { NVortexProblem } from './pkg/vortex_dynamics.js';

async function run() {
    const wasm = await init();
    var canvas;
    var ctx;
    var vortices;
    // Hold canvas information
    var WIDTH;
    var HEIGHT;
    var RADIUS;
     // how often, in milliseconds, we check to see if a redraw is needed
    var INTERVAL = 20;
    var isDrag = false;
    var isResizeDrag = false;
    var mx, my; // mouse coordinates
     // when set to true, the canvas will redraw everything
     // canvasValid = false; just sets this to false right now
     // we want to call canvasValid = false; whenever we make a change
    var canvasValid = false;
    // The node (if any) being selected.
    // If in the future we want to select multiple objects, this will get turned into an array
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
    // Context menu
    const menu = document.querySelector(".contextmenu");
    let menuVisible = false;

    function Vortex() {
      this.x = 0;
      this.y = 0;
      this.gamma = 1;
      this.fill = '#444444';
    }

    Vortex.prototype = {
      draw: function(context, optionalColor) {
        if (context === gctx) {
          context.fillStyle = 'black'; // always want black for the ghost canvas
        } else {
          context.fillStyle = this.fill;
        }
        // We can skip the drawing of elements that have moved off the screen:
        if (this.x > WIDTH || this.y > HEIGHT) return;
        if (this.x + this.w < 0 || this.y + this.h < 0) return;
        // When selected (or hovered), the vortex receives an extra shadow
        if (mySel === this || myHover === this) {
          context.beginPath();
          var coords = toScreenCoordinates(this.x, this.y);
          context.arc(coords[0], coords[1], 4, 0, 2 * Math.PI, false);
          context.strokeStyle = "gray";
          context.lineWidth = 3;
          context.stroke();
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
      const controls = document.getElementById("controls");
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
      div.appendChild(slider);
      div.appendChild(label);
      document.getElementById("controls").appendChild(div);
    }

    function gammaExpr(slider) {
      var value = parseFloat(slider.value).toFixed(2);
      return `\\(\\Gamma_${slider.vortex.index + 1}=${value}\\)`
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
      console.log("Draw");
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
      if (selectVortex(e)) { canvasValid = false;; }
      if (isDrag) {
        getMouse(e);
        var math_coords = toMathCoordinates(mx - offsetx, my - offsety);
        mySel.x = math_coords[0];
        mySel.y = math_coords[1];
        canvasValid = false;;
      }
      getMouse(e);
      if (mySel !== null) {
        this.style.cursor = 'auto';
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
          left: e.pageX,
          top: e.pageY
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
      // add three starting vortices
      addVortex(-1/2, 1/4, -.66);
      addVortex(2/3, 1/2, .66);
      addVortex(0, 0, -.25);
    }

    function togglePause() {
      if (paused) {
        // 
      }
      paused = !paused;
      mainLoop();
    }

    function mainLoop() {
      if (paused) {
        // make mainDraw() fire every INTERVAL milliseconds
        setInterval(mainDraw, INTERVAL);
      } else {
        updateVortices();
        canvasValid = false;
        mainDraw();
        requestAnimationFrame(mainLoop);
      }
    }

    initDisplay();
    mainLoop();
}

run();
