import React, { useState } from 'react';
import { StyleSheet, View, Text, TextInput, TouchableOpacity, FlatList, KeyboardAvoidingView, Platform } from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { FadeInUp, FadeInDown } from 'react-native-reanimated';

interface Message {
    id: string;
    text: string;
    sender: 'user' | 'ai';
}

export default function ChatScreen() {
    const [messages, setMessages] = useState<Message[]>([
        { id: '1', text: 'Hello! I am your personal financial assistant. How can I help you today?', sender: 'ai' },
    ]);
    const [inputText, setInputText] = useState('');

    const sendMessage = () => {
        if (!inputText.trim()) return;

        const userMsg: Message = { id: Date.now().toString(), text: inputText, sender: 'user' };
        setMessages(prev => [...prev, userMsg]);
        setInputText('');

        // Simulate AI response
        setTimeout(() => {
            const aiMsg: Message = {
                id: (Date.now() + 1).toString(),
                text: 'I can help you analyze your spending, set budgets, or transfer money. Just ask!',
                sender: 'ai'
            };
            setMessages(prev => [...prev, aiMsg]);
        }, 1000);
    };

    const renderItem = ({ item }: { item: Message }) => (
        <Animated.View
            entering={FadeInUp.duration(400)}
            style={[
                styles.messageBubble,
                item.sender === 'user' ? styles.userBubble : styles.aiBubble
            ]}
        >
            <Text style={[
                styles.messageText,
                item.sender === 'user' ? styles.userText : styles.aiText
            ]}>{item.text}</Text>
        </Animated.View>
    );

    return (
        <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={styles.container}
        >
            <View style={styles.header}>
                <Text style={styles.headerTitle}>Arthya AI</Text>
            </View>

            <FlatList
                data={messages}
                renderItem={renderItem}
                keyExtractor={item => item.id}
                contentContainerStyle={styles.listContent}
            />

            <View style={styles.inputContainer}>
                <TextInput
                    style={styles.input}
                    value={inputText}
                    onChangeText={setInputText}
                    placeholder="Ask anything..."
                    placeholderTextColor={Colors.textDim}
                />
                <TouchableOpacity onPress={sendMessage} style={styles.sendButton}>
                    <Ionicons name="send" size={20} color="#000" />
                </TouchableOpacity>
            </View>
        </KeyboardAvoidingView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.background,
    },
    header: {
        height: 60,
        justifyContent: 'center',
        alignItems: 'center',
        borderBottomWidth: 1,
        borderBottomColor: Colors.border,
        marginTop: 40,
        backgroundColor: Colors.background,
    },
    headerTitle: {
        color: Colors.text,
        fontSize: 18,
        fontWeight: '600',
    },
    listContent: {
        padding: 20,
    },
    messageBubble: {
        maxWidth: '80%',
        padding: 12,
        borderRadius: 20,
        marginBottom: 12,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 3,
        elevation: 2,
    },
    userBubble: {
        alignSelf: 'flex-end',
        backgroundColor: Colors.primary,
        borderBottomRightRadius: 4,
    },
    aiBubble: {
        alignSelf: 'flex-start',
        backgroundColor: Colors.card,
        borderBottomLeftRadius: 4,
        borderWidth: 1,
        borderColor: Colors.border,
    },
    messageText: {
        fontSize: 16,
    },
    userText: {
        color: '#000',
    },
    aiText: {
        color: Colors.text,
    },
    inputContainer: {
        flexDirection: 'row',
        padding: 16,
        borderTopWidth: 1,
        borderTopColor: Colors.border,
        alignItems: 'center',
        backgroundColor: Colors.background,
    },
    input: {
        flex: 1,
        backgroundColor: Colors.card,
        borderRadius: 24,
        paddingHorizontal: 20,
        paddingVertical: 12,
        color: Colors.text,
        marginRight: 12,
        borderWidth: 1,
        borderColor: Colors.border,
        fontSize: 16,
    },
    sendButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: Colors.primary,
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.3,
        shadowRadius: 5,
        elevation: 5,
    },
});
