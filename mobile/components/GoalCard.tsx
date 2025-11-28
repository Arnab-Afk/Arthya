import React from 'react';
import { StyleSheet, View, Text } from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';

interface GoalCardProps {
    title: string;
    currentAmount: string;
    targetAmount: string;
    progress: number; // 0 to 1
    icon: keyof typeof Ionicons.glyphMap;
}

export function GoalCard({ title, currentAmount, targetAmount, progress, icon }: GoalCardProps) {
    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <View style={styles.iconContainer}>
                    <Ionicons name={icon} size={20} color={Colors.text} />
                </View>
                <Text style={styles.percentage}>{Math.round(progress * 100)}%</Text>
            </View>

            <Text style={styles.title}>{title}</Text>
            <Text style={styles.amount}>{currentAmount} / <Text style={styles.target}>{targetAmount}</Text></Text>

            <View style={styles.progressBarBg}>
                <View style={[styles.progressBarFill, { width: `${progress * 100}%` }]} />
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        backgroundColor: Colors.card,
        borderRadius: 24,
        padding: 20,
        marginRight: 16,
        width: 160,
        borderWidth: 1,
        borderColor: Colors.border,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 6,
        elevation: 4,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 16,
    },
    iconContainer: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: Colors.cardElevated,
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 1,
        borderColor: Colors.border,
    },
    percentage: {
        color: Colors.primary,
        fontWeight: 'bold',
        fontSize: 14,
    },
    title: {
        color: Colors.text,
        fontSize: 16,
        fontWeight: '600',
        marginBottom: 4,
    },
    amount: {
        color: Colors.text,
        fontSize: 12,
        fontWeight: 'bold',
        marginBottom: 12,
    },
    target: {
        color: Colors.textDim,
        fontWeight: 'normal',
    },
    progressBarBg: {
        height: 6,
        backgroundColor: '#333',
        borderRadius: 3,
        overflow: 'hidden',
    },
    progressBarFill: {
        height: '100%',
        backgroundColor: Colors.primary,
        borderRadius: 3,
    },
});
