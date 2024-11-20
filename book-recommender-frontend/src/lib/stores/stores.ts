import { writable } from "svelte/store";
import { browser } from '$app/environment'

// Типот на податоците кои ќе ги чуваме во localStorage
interface Ratings {
    [bookId: number]: number;
}
let storedRatings = "{}"

// Svelte може да се извршува на страна на серверот или на страна на прелистувачот
// ако сме во прелистувач, прочитај ги податоците од localStorage
if(browser){
    storedRatings = localStorage.getItem('bookRatings') || '{}';
}
// Store за чување на рејтинзи
export const ratingsStore = writable<Ratings>(storedRatings ? JSON.parse(storedRatings) : {});

// Ако сме на страна на прелистувачот, запиши ги промените во localStorage
if (browser) {
    ratingsStore.subscribe((value) => {
        localStorage.setItem('bookRatings', JSON.stringify(value));
    });
}

// функција за додавање или ажурирање на рејтинг за дадена книга
export function updateRating(bookId: number, rating: number) {
    ratingsStore.update((currentRatings) => {
        
        return { ...currentRatings, [bookId]: rating };
    });
}

// функција за бришење на рејтинг за дадена книга
export function removeRating(bookId: number) {
    ratingsStore.update((currentRatings) => {
        const { [bookId]: _, ...remainingRatings } = currentRatings;
        return remainingRatings;
    });
}

// функција за добивање на рејтинг за дадена книга
export function getRating(bookId:number){
    let currentRatings: Ratings = {};
    ratingsStore.subscribe(value => {
        currentRatings = value;
    })();
    return currentRatings[bookId];
    
}

// Store за чување на препораки
export const recommendationsStore = writable([]);