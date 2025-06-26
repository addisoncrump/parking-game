// Copyright (c) Addison Crump, 2025, licensed under the EUPL-1.2-or-later.

//! parking-game: a library recreating the rules of Thinkfun's "Rush Hour".
//!
//! This library implements the core movement rules of "Rush Hour", with a focus on memory and
//! performance. The premise of the game is simple: you have one to many "cars" with fixed
//! orientations (up/down or left/right) that can only move forwards and backwards in that
//! orientation. You must move the designated car from its given start position to a desired end
//! position by manipulating the other cars in the board. Cars may not intersect and they must stay
//! within the bounds of the board. In this library, we only implement the _movement_ rules
//! (intersection and bounds checks included); the gameplay is left to the user.

#![no_std]

use alloc::vec;
use alloc::vec::Vec;
use core::error::Error;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::num::{IntErrorKind, NonZeroUsize};
use core::ops::{Add, AddAssign, Deref, DerefMut, Neg, Sub, SubAssign};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, One, Unsigned, Zero};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

extern crate alloc;

/// An orientation for a car.
#[derive(Copy, Clone, Debug, Hash, Deserialize, Serialize)]
pub enum Orientation {
    /// The car may only move up and down.
    UpDown,
    /// The car may only move left and right.
    LeftRight,
}

/// A direction for a move. A direction may be flipped with [`Neg`] (i.e. `-`).
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Deserialize, Serialize)]
pub enum Direction {
    /// Upward movement.
    Up,
    /// Downward movement.
    Down,
    /// Leftward movement.
    Left,
    /// Rightward movement.
    Right,
}

impl Display for Direction {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Direction::Up => "up",
            Direction::Down => "down",
            Direction::Left => "left",
            Direction::Right => "right",
        })
    }
}

impl Neg for Direction {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

/// Marker trait: specifies that a value may be used for board definitions.
pub trait BoardValue:
    One
    + Ord
    + Add<Output = Self>
    + CheckedAdd
    + Sub<Output = Self>
    + CheckedSub
    + AddAssign
    + SubAssign
    + Copy
    + Into<usize>
    + TryFrom<usize>
    + Zero
    + CheckedMul
    + Debug
    + Display
    + Unsigned
    + DeserializeOwned
    + Serialize
    + 'static
{
}

impl<V> BoardValue for V where
    V: One
        + Ord
        + Add<Output = Self>
        + CheckedAdd
        + Sub<Output = Self>
        + CheckedSub
        + AddAssign
        + SubAssign
        + Copy
        + Into<usize>
        + TryFrom<usize>
        + Zero
        + CheckedMul
        + Debug
        + Display
        + Unsigned
        + DeserializeOwned
        + Serialize
        + 'static
{
}

/// A car, generic over the numeric type which backs it. The numeric type must be unsigned and
/// integral.
#[derive(Copy, Clone, Debug, Hash, Deserialize, Serialize)]
pub struct Car<V> {
    length: V,
    orientation: Orientation,
}

impl<V> Car<V> {
    /// The length of the car.
    pub fn length(&self) -> &V {
        &self.length
    }

    /// The orientation of the car.
    pub fn orientation(&self) -> Orientation {
        self.orientation
    }
}

impl<V> Car<V>
where
    V: BoardValue,
{
    /// Create a new car of the provided length and orientation.
    pub fn new(length: V, orientation: Orientation) -> Option<Self> {
        if length < V::one() {
            None
        } else {
            Some(Self {
                length,
                orientation,
            })
        }
    }
}

/// A position in the board (eff., a coordinate pair).
#[derive(Copy, Clone, Debug, Hash, Deserialize, Serialize)]
pub struct Position<V> {
    row: V,
    column: V,
}

impl<V> Position<V> {
    /// The row of the position.
    pub fn row(&self) -> &V {
        &self.row
    }

    /// The column of the position.
    pub fn column(&self) -> &V {
        &self.column
    }
}

impl<V> Add for Position<V>
where
    V: BoardValue,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            row: self.row + rhs.row,
            column: self.column + rhs.column,
        }
    }
}

impl<V> CheckedAdd for Position<V>
where
    V: BoardValue,
{
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(Self {
            row: self.row.checked_add(&rhs.row)?,
            column: self.column.checked_add(&rhs.column)?,
        })
    }
}

impl<V> Sub for Position<V>
where
    V: BoardValue,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            row: self.row - rhs.row,
            column: self.column - rhs.column,
        }
    }
}

impl<V> CheckedSub for Position<V>
where
    V: BoardValue,
{
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(Self {
            row: self.row.checked_sub(&rhs.row)?,
            column: self.column.checked_sub(&rhs.column)?,
        })
    }
}

impl<V> AddAssign for Position<V>
where
    V: BoardValue,
{
    fn add_assign(&mut self, rhs: Self) {
        self.row += rhs.row;
        self.column += rhs.column;
    }
}

impl<V> Position<V>
where
    V: BoardValue,
{
    /// The position encoded as an index into an board with the provided dimensions.
    pub fn as_index(&self, dim: &Dimensions<V>) -> Option<usize> {
        if self.row >= dim.rows || self.column >= dim.columns {
            return None;
        }
        let row = self.row.into();
        let column = self.column.into();
        Some(row * dim.columns.into() + column)
    }
}

impl<V> Position<V>
where
    Self: CheckedAdd<Output = Self> + CheckedSub<Output = Self>,
    V: BoardValue,
{
    /// Get the position `by` units away from this position in the provided direction `dir`, or
    /// `None` if the position would be out of bounds.
    pub fn shift(&self, dir: Direction, by: V) -> Option<Self> {
        match dir {
            Direction::Up => self.checked_sub(&Self::from((by, V::zero()))),
            Direction::Down => self.checked_add(&Self::from((by, V::zero()))),
            Direction::Left => self.checked_sub(&Self::from((V::zero(), by))),
            Direction::Right => self.checked_add(&Self::from((V::zero(), by))),
        }
    }
}

impl<V> From<(V, V)> for Position<V> {
    fn from((row, column): (V, V)) -> Self {
        Self { row, column }
    }
}

/// The dimensions of a parking game board in terms of rows and columns.
#[derive(Copy, Clone, Debug, Hash, Deserialize, Serialize)]
pub struct Dimensions<V> {
    rows: V,
    columns: V,
}

impl<V> Dimensions<V> {
    /// The number of rows.
    pub fn rows(&self) -> &V {
        &self.rows
    }

    /// The number of columns.
    pub fn columns(&self) -> &V {
        &self.columns
    }
}

/// An error associated with the creation of the dimensions.
#[derive(Debug)]
pub struct DimensionError(IntErrorKind);

impl Display for DimensionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let reason = match self.0 {
            IntErrorKind::PosOverflow => "the dimensions were too large",
            IntErrorKind::Zero => "the dimensions have zero area",
            _ => unreachable!(),
        };
        f.write_fmt(format_args!("dimensions could not be used: {reason}"))
    }
}

impl Error for DimensionError {}

impl<V> TryFrom<(V, V)> for Dimensions<V>
where
    V: BoardValue,
{
    type Error = DimensionError;

    fn try_from((rows, columns): (V, V)) -> Result<Self, Self::Error> {
        if let Some(size) = rows.checked_mul(&columns) {
            if size.is_zero() {
                Err(DimensionError(IntErrorKind::Zero))
            } else {
                Ok(Self { rows, columns })
            }
        } else {
            Err(DimensionError(IntErrorKind::PosOverflow))
        }
    }
}

/// A state of the game. This is guaranteed to be a valid state as long as it is constructed with
/// [`State::empty`] and manipulated with via [`Board`] operations.
#[derive(Clone, Debug, Hash, Deserialize, Serialize)]
pub struct State<V> {
    dim: Dimensions<V>,
    cars: Vec<(Position<V>, Car<V>)>,
}

impl<V> State<V> {
    /// The dimensions of this state.
    pub fn dimensions(&self) -> &Dimensions<V> {
        &self.dim
    }

    /// The cars contained within this state.
    pub fn cars(&self) -> &[(Position<V>, Car<V>)] {
        &self.cars
    }
}

impl<V> State<V> {
    /// Produce an empty state (i.e., one with no cars) with the provided dimensions.
    pub fn empty<D: TryInto<Dimensions<V>>>(dim: D) -> Result<Self, D::Error> {
        let dim = dim.try_into()?;
        Ok(Self {
            dim,
            cars: Vec::new(),
        })
    }
}

/// A type of invalid state, associated with an [`InvalidStateError`].
#[derive(Debug)]
pub enum InvalidStateType {
    /// The car with the provided index is in an invalid position.
    InvalidPosition(NonZeroUsize),
    /// The cars with the provided indices overlap.
    Overlap(NonZeroUsize, NonZeroUsize),
}

/// An error which denotes that an invalid state was encountered.
#[derive(Debug)]
pub struct InvalidStateError<V> {
    position: Position<V>,
    variant: InvalidStateType,
}

impl<V> Display for InvalidStateError<V>
where
    V: BoardValue,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self.variant {
            InvalidStateType::InvalidPosition(idx) => f.write_fmt(format_args!(
                "car {idx} was located at an invalid position ({}, {})",
                self.position.row, self.position.column
            )),
            InvalidStateType::Overlap(idx1, idx2) => f.write_fmt(format_args!(
                "car {idx1} and car {idx2} overlapped at position ({}, {})",
                self.position.row, self.position.column
            )),
        }
    }
}

impl<V> Error for InvalidStateError<V> where V: BoardValue {}

fn add_car_concrete<V>(
    board: &mut [Option<NonZeroUsize>],
    idx: NonZeroUsize,
    dim: &Dimensions<V>,
    position: &Position<V>,
    car: &Car<V>,
) -> Result<(), InvalidStateError<V>>
where
    V: BoardValue,
{
    let mut base = *position;
    let offset = match car.orientation {
        Orientation::UpDown => Position {
            row: V::one(),
            column: V::zero(),
        },
        Orientation::LeftRight => Position {
            row: V::zero(),
            column: V::one(),
        },
    };
    for _ in 0..car.length.into() {
        match base.as_index(dim).and_then(|p| board.get_mut(p)) {
            None => {
                return Err(InvalidStateError {
                    position: base,
                    variant: InvalidStateType::InvalidPosition(idx),
                });
            }
            Some(entry) => {
                if let Some(existing) = entry {
                    return Err(InvalidStateError {
                        position: base,
                        variant: InvalidStateType::Overlap(*existing, idx),
                    });
                }
            }
        }
        base += offset;
    }
    let mut base = *position;
    for _ in 0..car.length.into() {
        board[base.as_index(dim).unwrap()] = Some(idx);
        base += offset;
    }
    Ok(())
}

impl<V> State<V>
where
    V: BoardValue,
{
    fn concrete(&self) -> Result<Vec<Option<NonZeroUsize>>, InvalidStateError<V>> {
        let mut board = vec![None; self.dim.columns.into() * self.dim.rows.into()];
        for (idx, (position, car)) in self.cars.iter().enumerate() {
            add_car_concrete(
                &mut board,
                NonZeroUsize::new(idx + 1).unwrap(),
                &self.dim,
                position,
                car,
            )?;
        }
        Ok(board)
    }

    /// An immutable representation of the current board, or an error if this state is invalid.
    pub fn board(&self) -> Result<Board<&Self, V>, InvalidStateError<V>> {
        Ok(Board {
            concrete: self.concrete()?,
            state: self,
            phantom: PhantomData,
        })
    }

    /// A mutable representation of the current board, or an error if this state is invalid.
    pub fn board_mut(&mut self) -> Result<Board<&mut Self, V>, InvalidStateError<V>> {
        Ok(Board {
            concrete: self.concrete()?,
            state: self,
            phantom: PhantomData,
        })
    }
}

/// A concretised representation of the board.
#[derive(Debug)]
pub struct Board<R, V> {
    state: R,
    concrete: Vec<Option<NonZeroUsize>>,
    phantom: PhantomData<V>,
}

impl<R, V> Hash for Board<R, V>
where
    R: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state.hash(state);
    }
}

impl<R, V> Board<R, V> {
    /// The [`Vec`] which represents the board literally.
    pub fn concrete(&self) -> &Vec<Option<NonZeroUsize>> {
        &self.concrete
    }
}

impl<R, V> Board<R, V>
where
    R: Deref<Target = State<V>>,
    V: BoardValue,
{
    /// Fetches the car index occupying the requested position. [`None`] if the position doesn't
    /// exist in the board, [`Some`]`(`[`None`]`)` if the position exists, but is empty, and
    /// [`Some`]`(`[`Some`]`(n))` with `n` as the car that occupies that position.
    pub fn get<P: Into<Position<V>>>(&self, position: P) -> Option<Option<NonZeroUsize>> {
        position
            .into()
            .as_index(&self.state.dim)
            .and_then(|p| self.concrete.get(p).copied())
    }
}

impl<R, V> Board<R, V>
where
    R: Deref<Target = State<V>>,
{
    /// Gets the current state of the board.
    pub fn state(&self) -> &State<V> {
        self.state.deref()
    }
}

impl<R, V> Board<R, V>
where
    R: DerefMut<Target = State<V>>,
{
    /// Gets the current state of the board, mutably.
    pub fn state_mut(&mut self) -> &mut State<V> {
        self.state.deref_mut()
    }
}

/// The type of invalid move that was observed in an [`InvalidMoveError`].
#[derive(Debug)]
pub enum InvalidMoveType<V> {
    /// The car that was designated to be moved didn't exist.
    InvalidCar,
    /// The direction that was used isn't valid for the provided car.
    InvalidDirection,
    /// The final position of the car is out of bounds.
    InvalidFinalPosition,
    /// After moving the car, the car would intersect another with the provided index at the
    /// provided position.
    Intersects(Position<V>, NonZeroUsize),
}

/// An error which describes an attempted invalid move.
#[derive(Debug)]
pub struct InvalidMoveError<V> {
    car: NonZeroUsize,
    dir: Direction,
    variant: InvalidMoveType<V>,
}

impl<V> Display for InvalidMoveError<V>
where
    V: BoardValue,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match &self.variant {
            InvalidMoveType::InvalidCar => f.write_fmt(format_args!("cannot move car {} {} because it doesn't exist", self.car, self.dir)),
            InvalidMoveType::InvalidDirection => f.write_fmt(format_args!("cannot move car {} {} because its orientation does not allow for movement in that direction", self.car, self.dir)),
            InvalidMoveType::InvalidFinalPosition => f.write_fmt(format_args!("cannot move car {} {} because it enters an invalid position", self.car, self.dir)),
            InvalidMoveType::Intersects(pos, other) => f.write_fmt(format_args!("cannot move car {} {} because it would intersect with car {} at ({}, {})", self.car, self.dir, other, pos.row, pos.column)),
        }
    }
}

impl<V> Error for InvalidMoveError<V> where V: BoardValue {}

impl<R, V> Board<R, V>
where
    R: DerefMut<Target = State<V>>,
    V: BoardValue,
{
    /// Add a car to the board, updating the backing state in the process.
    pub fn add_car<P: Into<Position<V>>>(
        &mut self,
        position: P,
        car: Car<V>,
    ) -> Result<NonZeroUsize, InvalidStateError<V>> {
        let position = position.into();
        let idx = NonZeroUsize::new(self.state.cars.len() + 1).unwrap();
        add_car_concrete(&mut self.concrete, idx, &self.state.dim, &position, &car)?;
        self.state.cars.push((position, car));
        Ok(idx)
    }
}

impl<R, V> Board<R, V>
where
    R: DerefMut<Target = State<V>>,
    V: BoardValue,
{
    /// Shift a provided car one space in the designated direction.
    pub fn shift_car(
        &mut self,
        car: NonZeroUsize,
        dir: Direction,
    ) -> Result<Position<V>, InvalidMoveError<V>> {
        let idx = car.get() - 1;
        if let Some((pos, actual)) = self.state.cars.get(idx).copied() {
            let (deleted, inserted) = match (dir, actual.orientation) {
                (Direction::Up, Orientation::UpDown)
                | (Direction::Left, Orientation::LeftRight) => (
                    pos.shift(-dir, actual.length - V::one()),
                    pos.shift(dir, V::one()),
                ),
                (Direction::Down, Orientation::UpDown)
                | (Direction::Right, Orientation::LeftRight) => {
                    (Some(pos), pos.shift(dir, actual.length))
                }
                _ => {
                    return Err(InvalidMoveError {
                        car,
                        dir,
                        variant: InvalidMoveType::InvalidDirection,
                    });
                }
            };
            if let (Some(deleted_pos), Some(inserted_pos)) = (deleted, inserted) {
                let deleted =
                    deleted_pos
                        .as_index(&self.state.dim)
                        .ok_or(InvalidMoveError {
                            car,
                            dir,
                            variant: InvalidMoveType::InvalidFinalPosition,
                        })?;
                let inserted =
                    inserted_pos
                        .as_index(&self.state.dim)
                        .ok_or(InvalidMoveError {
                            car,
                            dir,
                            variant: InvalidMoveType::InvalidFinalPosition,
                        })?;
                if let Ok([deleted, inserted]) = self.concrete.get_disjoint_mut([deleted, inserted])
                {
                    return if let Some(idx) = inserted {
                        Err(InvalidMoveError {
                            car,
                            dir,
                            variant: InvalidMoveType::Intersects(inserted_pos, *idx),
                        })
                    } else {
                        *inserted = deleted.take();
                        let new = pos.shift(dir, V::one()).unwrap();
                        self.state.cars[idx].0 = new;
                        Ok(new)
                    };
                }
            }
            Err(InvalidMoveError {
                car,
                dir,
                variant: InvalidMoveType::InvalidFinalPosition,
            })
        } else {
            Err(InvalidMoveError {
                car,
                dir,
                variant: InvalidMoveType::InvalidCar,
            })
        }
    }
}

impl<R, V> Display for Board<R, V>
where
    R: Deref<Target = State<V>>,
    V: BoardValue,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let num_width = (self.state.cars.len().saturating_sub(1).max(1).ilog10() + 2)
            .next_multiple_of(2) as usize
            - 1;
        let side = num_width + 2;
        let mut indices = self.concrete.iter().copied();
        for _row in 0..self.state.dim.rows.into() {
            for _padding in 0..(side / 2) {
                writeln!(f, "{:#>1$}", "", self.state.dim.columns.into() * side)?;
            }
            for _column in 0..self.state.dim.columns.into() {
                write!(f, "#")?;
                if let Some(idx) = indices.next().unwrap() {
                    write!(f, "{idx:0num_width$}")?;
                } else {
                    write!(f, "{:#>num_width$}", "")?;
                }
                write!(f, "#")?;
            }
            writeln!(f)?;
            for _padding in 0..(side / 2) {
                writeln!(f, "{:#>1$}", "", self.state.dim.columns.into() * side)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    extern crate std;

    use crate::{
        Car, Direction, InvalidMoveError, InvalidMoveType, InvalidStateError, InvalidStateType,
        Orientation, State,
    };
    use alloc::boxed::Box;
    use core::error::Error;
    use core::num::NonZeroUsize;
    use std::println;

    #[test]
    fn simple_board() -> Result<(), Box<dyn Error>> {
        let mut state = State::empty((2u8, 2))?;
        let mut board = state.board_mut()?;
        println!("{board}");
        let idx = board.add_car((0, 0), Car::new(1, Orientation::LeftRight).unwrap())?;
        assert_eq!(1, idx.get());
        println!("{board}");
        assert_eq!(1, board.get((0, 0)).unwrap().unwrap().get());

        match board.add_car((2, 0), Car::new(1, Orientation::UpDown).unwrap()) {
            Err(InvalidStateError {
                position,
                variant: InvalidStateType::InvalidPosition(idx),
            }) => {
                assert_eq!((2, 0), (position.row, position.column));
                assert_eq!(2, idx.get());
            }
            _ => unreachable!("Should error here"),
        }

        match board.add_car((0, 0), Car::new(1, Orientation::UpDown).unwrap()) {
            Err(InvalidStateError {
                position,
                variant: InvalidStateType::Overlap(idx1, idx2),
            }) => {
                assert_eq!((0, 0), (position.row, position.column));
                assert_eq!(1, idx1.get());
                assert_eq!(2, idx2.get());
            }
            _ => unreachable!("Should error here"),
        }

        board.shift_car(idx, Direction::Right)?;
        assert_eq!(1, board.get((0, 1)).unwrap().unwrap().get());
        assert_eq!(None, board.get((0, 0)).unwrap());
        println!("{board}");

        board.shift_car(idx, Direction::Left)?;
        assert_eq!(1, board.get((0, 0)).unwrap().unwrap().get());
        assert_eq!(None, board.get((0, 1)).unwrap());
        println!("{board}");

        let idx = board.add_car((0, 1), Car::new(1, Orientation::UpDown).unwrap())?;
        assert_eq!(2, idx.get());
        println!("{board}");
        assert_eq!(2, board.get((0, 1)).unwrap().unwrap().get());

        board.shift_car(idx, Direction::Down)?;
        assert_eq!(2, board.get((1, 1)).unwrap().unwrap().get());
        assert_eq!(None, board.get((0, 1)).unwrap());
        println!("{board}");

        board.shift_car(idx, Direction::Up)?;
        assert_eq!(2, board.get((0, 1)).unwrap().unwrap().get());
        assert_eq!(None, board.get((1, 1)).unwrap());
        println!("{board}");

        match board.shift_car(NonZeroUsize::new(3).unwrap(), Direction::Right) {
            Err(InvalidMoveError {
                variant: InvalidMoveType::InvalidCar,
                car,
                dir,
            }) => {
                assert_eq!(3, car.get());
                assert_eq!(dir, Direction::Right)
            }
            s => unreachable!("Expected another error, got {s:?}"),
        }

        match board.shift_car(idx, Direction::Up) {
            Err(InvalidMoveError {
                variant: InvalidMoveType::InvalidFinalPosition,
                car,
                dir,
            }) => {
                assert_eq!(2, car.get());
                assert_eq!(dir, Direction::Up)
            }
            s => unreachable!("Expected another error, got {s:?}"),
        }

        match board.shift_car(idx, Direction::Right) {
            Err(InvalidMoveError {
                variant: InvalidMoveType::InvalidDirection,
                car,
                dir,
            }) => {
                assert_eq!(2, car.get());
                assert_eq!(dir, Direction::Right)
            }
            s => unreachable!("Expected another error, got {s:?}"),
        }

        match board.shift_car(NonZeroUsize::new(1).unwrap(), Direction::Right) {
            Err(InvalidMoveError {
                variant: InvalidMoveType::Intersects(at, with),
                car,
                dir,
            }) => {
                assert_eq!(1, car.get());
                assert_eq!(2, with.get());
                assert_eq!((0, 1), (at.row, at.column));
                assert_eq!(dir, Direction::Right);
            }
            s => unreachable!("Expected another error, got {s:?}"),
        }

        let concrete = board.concrete().clone();
        drop(board);

        let board = state.board()?;

        assert_eq!(&concrete, board.concrete());

        Ok(())
    }

    #[test]
    fn multi_board() -> Result<(), Box<dyn Error>> {
        let mut state = State::empty((5u8, 5))?;
        let mut board = state.board_mut()?;
        println!("{board}");
        let idx = board.add_car((0, 0), Car::new(3, Orientation::UpDown).unwrap())?;
        assert_eq!(1, idx.get());
        println!("{board}");

        board.shift_car(idx, Direction::Down)?;
        println!("{board}");

        Ok(())
    }
}
