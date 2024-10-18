import { writable } from "svelte/store";
import { browser } from '$app/environment'
// Define the type for the ratings store
interface Ratings {
    [bookId: number]: number;
}
let storedRatings = "{}"
// Initialize with an empty object to store ratings as { bookId: rating }
if(browser){

    storedRatings = localStorage.getItem('bookRatings') || '{}';
}

export const ratingsStore = writable<Ratings>(storedRatings ? JSON.parse(storedRatings) : {});

if (browser) {
    ratingsStore.subscribe((value) => {
        localStorage.setItem('bookRatings', JSON.stringify(value));
    });
}

// Function to update the rating of a book
export function updateRating(bookId: number, rating: number) {
    ratingsStore.update((currentRatings) => {
        // Update or add the rating for the given bookId
        return { ...currentRatings, [bookId]: rating };
    });
}

// Optional: Function to remove a rating for a specific book
export function removeRating(bookId: number) {
    ratingsStore.update((currentRatings) => {
        const { [bookId]: _, ...remainingRatings } = currentRatings;
        return remainingRatings;
    });
}

export function getRating(bookId:number){
    let currentRatings: Ratings = {};
    ratingsStore.subscribe(value => {
        currentRatings = value;
    })();
    return currentRatings[bookId];
    
}