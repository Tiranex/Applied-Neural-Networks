const columns = 7;
const rows = 6;

const boardArray = Array.from({ length: rows }, () => Array(columns).fill(0));
const htmlArray = Array.from({ length: rows }, () => Array(columns).fill(0));

let game_over = false;
let prev_hover = null;
let plays = 0;

window.onload = function() {
    createBoard();
}

function reset(){
    for(let i=0; i<rows; i++){
        for(let j=0; j<columns; j++){
            boardArray[i][j] = 0;
            htmlArray[i][j].style.backgroundColor = 'white';
        }
    }
    game_over = false;
    plays = 0;
    const winner_h = document.getElementById('winner_tag');
    winner_h.textContent = '---';
}

function createBoard() {
    const board = document.getElementById('board');
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.id = i + '-' + j;
            cell.onclick = function() {
                if(dropPiece(i,j) == 0)
                    return;
                const action = predict();
                console.log(action)
                dropPiece(0, action)
            }
            cell.onmouseenter = function (){ hover(i,j) };
            cell.onmouseleave = function (){ hover_leave() }
            board.appendChild(cell);
            htmlArray[i][j] = cell;
        }
    }
}

let turn = -1;
let color ='red';
function change(){
    if(turn == -1){
        turn = 1;
        color='blue';
    }
    else{
        turn = -1;
        color='red';
    }
}



function dropPiece(i,j){
    if(game_over){
        return 0;
    }

    const column = j;
    const row = find_lowest_piece(column);

    if (row != null){
        boardArray[row][column] = turn;
        change();
        htmlArray[row][column].style.backgroundColor = color;
        prev_hover = null;
        plays++;
        if(checkWin()!=null)
            setWinner(checkWin());
        return 1;
    }
    return 0;
}

function setWinner(winner){
    const winner_h = document.getElementById('winner_tag');
    if(winner == -1){
        winner_h.textContent = 'Blue wins!';
    }
    else if(winner == 1){
        winner_h.textContent = 'Red wins!';
    }
    else{
        winner_h.textContent = 'It\'s a draw!';
    }

}

function find_lowest_piece(column){
    for(let i=rows-1; i>=0; i--){
        if(boardArray[i][column] == 0){
            return i;
        }
    }
    return null;
}

function checkWin(){
    // Horizontal
    for(let i=0; i<rows; i++){
        for(let j=0; j<columns-3; j++){
            if(boardArray[i][j] != 0){
                if(boardArray[i][j] === boardArray[i][j+1] && boardArray[i][j] === boardArray[i][j+2] && boardArray[i][j] === boardArray[i][j+3]){
                    game_over = true;
                    console.log('Winner: ', boardArray[i][j]);
                    return boardArray[i][j];
                }
            }
        }
    }

    // Vertical
    for(let i=0; i<rows-3; i++){
        for(let j=0; j<columns; j++){
            if(boardArray[i][j] != 0){
                if(boardArray[i][j] === boardArray[i+1][j] && boardArray[i][j] === boardArray[i+2][j] && boardArray[i][j] === boardArray[i+3][j]){
                    game_over = true;
                    console.log('Winner: ', boardArray[i][j]);
                    return boardArray[i][j];
                }
            }
        }
    }

    // Diagonal
    for(let i=0; i<rows-3; i++){
        for(let j=0; j<columns-3; j++){
            if(boardArray[i][j] != 0){
                if(boardArray[i][j] === boardArray[i+1][j+1] && boardArray[i][j] === boardArray[i+2][j+2] && boardArray[i][j] === boardArray[i+3][j+3]){
                    game_over = true;
                    console.log('Winner: ', boardArray[i][j]);
                    return boardArray[i][j];
                }
            }
        }
    }

    // Antidiagonal
    for(let i=0; i<rows-3; i++){
        for(let j=3; j<columns; j++){
            if(boardArray[i][j] != 0){
                if(boardArray[i][j] === boardArray[i+1][j-1] && boardArray[i][j] === boardArray[i+2][j-2] && boardArray[i][j] === boardArray[i+3][j-3]){
                    game_over = true;
                    console.log('Winner: ', boardArray[i][j]);
                    return boardArray[i][j];
                }
            }
        }
    }

    // Draw
    if(plays == rows*columns){
        game_over = true;
        return 0;
    }

    return null
}

function hover(i,j){
    const column = j;
    const row = find_lowest_piece(column);
    if (row != null && game_over == false){
        htmlArray[row][column].style.backgroundColor = 'gray';
        prev_hover = [row, column];
    }
}

function hover_leave(){
    if(prev_hover != null){
        htmlArray[prev_hover[0]][prev_hover[1]].style.backgroundColor = 'white';
        prev_hover=null;
    }
}

/* Model Prediction */
function predict(){
    let tensor = tf.tensor(boardArray);
    tensor = tensor.reshape([1, rows, columns, 1]);
    
    let prediction = model.predict(tensor).dataSync();
    console.log(prediction);

    return tf.argMax(prediction, axis=-1).dataSync()[0]
}

/* Model loading */
let model

async function loadModel() {
  model = await tf.loadLayersModel('model/model.json');
}

loadModel();